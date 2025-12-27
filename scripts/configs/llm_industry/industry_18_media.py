from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="18",
    label="Media",
    profile_id="industry:18@v1",
    presence_addendum=(
        "Offerings include media production, publishing, content services, and advertising offerings.\n"
        "Signals: studio services, production, post-production, publishing, subscriptions, channels, ad solutions.\n"
        "Ignore pure news articles unless they describe subscriptions/products."
    ),
    full_addendum=(
        "Products: subscriptions, publications, content libraries, media properties if monetized.\n"
        "Services: production, post-production, distribution, licensing, advertising solutions, creative services.\n"
        "If they offer 'platform' as ad-tech/media platform, treat as service unless sold as standalone software product.\n"
        "Tags: 'production', 'post-production', 'publishing', 'subscription', 'advertising', 'licensing'."
    ),
)
