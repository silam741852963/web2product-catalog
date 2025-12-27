from __future__ import annotations
from .base import IndustryLLMProfile

PROFILE = IndustryLLMProfile(
    code="17",
    label="Hospitality",
    profile_id="industry:17@v1",
    presence_addendum=(
        "Offerings include accommodation, venues, packages, and hospitality services.\n"
        "Signals: rooms/suites, dining, events, weddings, packages, amenities, bookings, tours.\n"
        "Do not treat general location descriptions as offerings unless they map to services (e.g., 'banquet services')."
    ),
    full_addendum=(
        "Services: accommodation types, event venue services, catering/dining services, spa/wellness, tour packages if sold.\n"
        "Products: gift cards/merchandise only if explicitly sold.\n"
        "If the page lists room types, group into one offering like 'Hotel accommodation (room & suite categories)'.\n"
        "Tags: 'hotel', 'resort', 'events', 'weddings', 'catering', 'spa', 'packages'."
    ),
)
