"""Catalog + hard-constraint helpers. Reuse these so your recommender respects the property gate."""

from __future__ import annotations


def is_eligible(item: dict, attrs: dict) -> bool:
    """An item is eligible if it stocks the person's size and is within budget."""
    return attrs["size"] in item["sizes"] and item["price"] <= attrs["budget_usd"]


def eligible_items(catalog: list[dict], attrs: dict) -> list[dict]:
    return [it for it in catalog if is_eligible(it, attrs)]


def index_by_id(catalog: list[dict]) -> dict[str, dict]:
    return {it["id"]: it for it in catalog}
