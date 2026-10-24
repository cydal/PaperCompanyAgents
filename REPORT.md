# Beaver's Choice Paper Company — Multi-Agent System Report

## 1. System Overview

This project implements a 4-agent system: an Orchestrator plus three worker agents (Inventory, Quoting, Sales). They handle free-text customer requests
for paper supplies end-to-end, from parsing the request, checking and reordering
stock, pricing the order with bulk discounts, and finalizing or rejecting the
sale. It is built with **pydantic-ai** and implemented entirely in
[`project_starter.py`](project_starter.py). The workflow diagram is at
[`workflow_diagram.png`](workflow_diagram.png) (source:
[`workflow_diagram.mmd`](workflow_diagram.mmd)).

## 2. Architecture and Design Decisions

### 2.1 The four agents & Roles

We setup one orchestrator and
three workers agents: (inventory, quoting, sales) with no overlapping responsibilities:

- **Orchestrator Agent** — the only agent that sees the customer text. It
  parses free text into structured line items, and delegates to the three workers, it composes the
  final message.
- **Inventory Agent** — answers "do we have enough stock, and if not, should
  we reorder?" Owns `get_stock_level`, `get_all_inventory`,
  `get_supplier_delivery_date`, `get_cash_balance`, and `create_transaction` for `stock_orders`.
- **Quoting Agent** — answers "what should this cost?" Owns
  `search_quote_history` for discount calibration against similar past
  orders.
- **Sales Agent** — answers "can we actually commit to this?"
  Owns `generate_financial_report` and
  `create_transaction` for `sales`. Decides fulfill-or-reject and writes the reason when rejecting.

Helper functions:

| Helper function | Owning agent | Tool |
|---|---|---|
| `get_stock_level` | Inventory | `check_stock` |
| `get_supplier_delivery_date` | Inventory | `estimate_restock_delivery` |
| `get_cash_balance` | Inventory | `check_cash` |
| `create_transaction` (`stock_orders`) | Inventory | `reorder_stock` |
| `search_quote_history` | Quoting | `find_similar_quotes` |
| `generate_financial_report` | Sales | `run_financial_health_check` |
| `create_transaction` (`sales`) | Sales | `finalize_sale` |

(`get_all_inventory` is available for inventory snapshots)

### 2.2 A single delegation tool, by design

The Orchestrator delegates through exactly **one** tool, `process_order`,
rather than calling the Inventory, Quoting, and Sales agents as three
separate tool calls. `process_order` is a Python function that `await`s all three
worker agents in a fixed sequence (Inventory → Quoting → Sales) and returns
their combined outcome. 

### 2.3 Deterministic functions

Three decisions in the system are kept out of the LLM's hands entirely,
computed in plain Python, and handed to the agents as given facts:

1. **Pricing and discount tier** (`price_line_items`, `discount_tier`) — a
   fixed quantity-based schedule (≥1000 units → 15%, ≥500 → 10%, ≥100 → 5%,
   else 0%). 
2. **Delivery-deadline feasibility** — each item's `available_by`
   date is compared against the customer's requested delivery date using plain string
   comparison.

### 2.4 Catalog matching: a real failure found during testing

During initial testing, the Orchestrator mapped "100 sheets of heavy
cardstock" to the catalog item **"Heavyweight paper"** instead of the
obviously-correct **"Cardstock"** — the `Literal` constraint guaranteed a
*valid* item but not the *right* one. It also force-mapped "A3 paper" (which
isn't in the catalog at all) to the closest available item rather than
recognizing we simply don't carry it. Both were fixed by: (a) listing the
full 46-item catalog directly in the Orchestrator's system prompt with an
explicit instruction to match on the core product noun rather than
incidental adjectives, and (b) instructing the model to leave an item out of
the order (and tell the customer we don't carry it) rather than force a
mapping when there's no reasonable match. Section 4 shows this fix working
in the final run.


## 3. Framework and Implementation Notes

The project is developed with **pydantic-ai** with OpenAI (`gpt-4o-mini` by default,
configurable via `OPENAI_MODEL` in `.env`). LangSmith tracing (via OpenTelemetry) is
wired in and active for observability during development, enabled whenever
`LANGSMITH_API_KEY` is present in `.env`.

## 4. Evaluation Results

The system was run against all 20 requests in `quote_requests_sample.csv`,
producing [`test_results.csv`](test_results.csv). Summary:

- **10 of 20 requests fulfilled**, 10 rejected
- **Cash balance changed on 18 of 20 requests** 
- Every rejection includes a specific reason:
  - **Insufficient stock**: *"we currently have insufficient stock of
    cardstock and cannot provide it at an affordable price"* (6).
  - **Missed delivery deadline**: *"the cardstock cannot be delivered by the
    requested date of April 10, 2025; it will be available by April 12,
    2025"* (13)
  - **Item not carried**: *"we don't carry A3 paper or printer paper in our
    catalog. However, we can fulfill your request for A4 paper"* (3)

### Strengths

- **Transparency**: every response states what was ordered and the exact
  total and every rejection states the specific cause. 
- **Consistent pricing**: because discounting is deterministic rather than
  LLM-decided, the same order size always produces the same discount.

### Weaknesses observed
- The vague rejection message style "inventory availability is not fully
  confirmed".
- Outcomes for borderline orders can still vary slightly between runs.

## 5. Suggestions for Further Improvement

1. **Partial order fulfillment.** Currently, one unfulfillable line item
   rejects the entire order. A better approach would
   fulfill the items it can and only reject the specific item(s) it can't.