"""Seed data for the public demo (DEMO_MODE=true)."""
import secrets

from sqlmodel import Session, select

from app.core.security import generate_2fa_secret, hash_password
from app.models.domain import Bus, Route, User

DEMO_USER = "demo"
DEMO_PASSWORD = "demo-pass-2026"

ROUTES = [
    ("R-12", "Kashmere Gate - Nehru Place", "Kashmere Gate", "Nehru Place", 21.4),
    ("R-34", "Anand Vihar - Dwarka", "Anand Vihar", "Dwarka Sec 21", 32.8),
    ("R-56", "Noida Sec 62 - Rajiv Chowk", "Noida Sec 62", "Rajiv Chowk", 24.1),
]
BUSES = [("DL1PC4521", "R-12"), ("DL1PC7730", "R-34"), ("DL1PD1188", "R-56")]


def seed_demo(engine) -> None:
    with Session(engine) as session:
        if session.exec(select(User)).first():
            return
        # Taking the admin slot first stops a visitor from registering as admin.
        session.add(User(username="admin", role="admin", password_hash=hash_password(secrets.token_urlsafe(24)),
                         twofa_secret=generate_2fa_secret()))
        session.add(User(username=DEMO_USER, role="operator", password_hash=hash_password(DEMO_PASSWORD),
                         twofa_secret=generate_2fa_secret()))
        for route_id, name, start, end, km in ROUTES:
            session.add(Route(route_id=route_id, name=name, start_location=start, end_location=end, distance_km=km))
        for bus_id, route_id in BUSES:
            session.add(Bus(bus_id=bus_id, route_id=route_id, capacity=52, model="Tata Starbus CNG", year=2022))
        session.commit()
