"""
scripts/generate_data.py
========================
Generates synthetic training data for the Pocket-Agent tool-call fine-tune.

Usage:
    python scripts/generate_data.py --out data/train.jsonl --seed 42

Output: JSONL file where each line is:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Covers:
  - Single-turn tool calls for all 5 tools
  - Multi-turn conversations with reference resolution
  - Refusals (chitchat, impossible tools, ambiguous with no history)
  - Adversarial (typos, code-switched Urdu/Spanish/Arabic, unit ambiguity, hallucination-bait)

SHA-256 of training prompts is printed for grader verification.
"""

import argparse
import hashlib
import json
import os
import random
import sys


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Pocket-Agent, an on-device mobile assistant.\n"
    "You have access to 5 tools: weather, calendar, convert, currency, sql.\n\n"
    "For unambiguous requests, emit ONLY a JSON tool call wrapped in "
    "<tool_call>...</tool_call> tags.\n"
    "For chitchat, impossible tools, or ambiguous references with no prior history, "
    "emit plain text with NO tool call.\n"
    "Never guess — if the user's intent is unclear and there is no prior context, "
    "refuse politely.\n\n"
    "Tool schemas:\n"
    '{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}\n'
    '{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}\n'
    '{"tool": "convert",  "args": {"value": number, "from_unit": "string", "to_unit": "string"}}\n'
    '{"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}\n'
    '{"tool": "sql",      "args": {"query": "string"}}\n'
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def tc(tool: str, **args) -> str:
    """Build a <tool_call> string."""
    return f"<tool_call>{json.dumps({'tool': tool, 'args': args})}</tool_call>"


def example(user: str, assistant: str, history: list = None) -> dict:
    """Build a training example dict in the messages format."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user",      "content": user})
    msgs.append({"role": "assistant", "content": assistant})
    return {"messages": msgs}


def cur(amount, from_iso, to_iso):
    """Helper for currency tool calls (avoids Python reserved word 'from')."""
    return tc("currency", amount=amount, **{"from": from_iso, "to": to_iso})


# ── Example pools ─────────────────────────────────────────────────────────────

def weather_examples():
    return [
        example("What's the weather in London?",              tc("weather", location="London", unit="C")),
        example("Weather in New York in Fahrenheit",          tc("weather", location="New York", unit="F")),
        example("How's the weather in Tokyo right now?",      tc("weather", location="Tokyo", unit="C")),
        example("Is it cold in Berlin today?",                tc("weather", location="Berlin", unit="C")),
        example("Temperature in Dubai please",                tc("weather", location="Dubai", unit="C")),
        example("What's the temp in Chicago in F?",           tc("weather", location="Chicago", unit="F")),
        example("Tell me the weather in Paris",               tc("weather", location="Paris", unit="C")),
        example("Weather for Sydney Australia",               tc("weather", location="Sydney", unit="C")),
        example("Current conditions in Mumbai",               tc("weather", location="Mumbai", unit="C")),
        example("Weather in Moscow in Celsius",               tc("weather", location="Moscow", unit="C")),
        example("Check weather Los Angeles Fahrenheit",       tc("weather", location="Los Angeles", unit="F")),
        example("Weather update for Karachi",                 tc("weather", location="Karachi", unit="C")),
        example("How's the weather over in Berlin today?",    tc("weather", location="Berlin", unit="C")),
        example("Current temperature in Dubai",               tc("weather", location="Dubai", unit="C")),
        example("Clima en Buenos Aires",                      tc("weather", location="Buenos Aires", unit="C")),
        example("What is the temperature in Seoul?",          tc("weather", location="Seoul", unit="C")),
        example("Weather Singapore",                          tc("weather", location="Singapore", unit="C")),
        example("How hot is it in Cairo?",                    tc("weather", location="Cairo", unit="C")),
        example("Nairobi weather today",                      tc("weather", location="Nairobi", unit="C")),
        example("Give me the weather for Amsterdam",          tc("weather", location="Amsterdam", unit="C")),
    ]


def calendar_examples():
    return [
        example("Show my meetings for 2025-03-15",               tc("calendar", action="list", date="2025-03-15")),
        example("List my events on April 5 2025",                tc("calendar", action="list", date="2025-04-05")),
        example("Create a meeting called Team Standup on 2025-06-01", tc("calendar", action="create", date="2025-06-01", title="Team Standup")),
        example("Add dentist appointment on 2025-07-22",         tc("calendar", action="create", date="2025-07-22", title="Dentist appointment")),
        example("What's on my calendar for 2025-01-20?",         tc("calendar", action="list", date="2025-01-20")),
        example("Schedule a call with Ahmed on 2025-08-10",      tc("calendar", action="create", date="2025-08-10", title="Call with Ahmed")),
        example("List events 2025-12-25",                        tc("calendar", action="list", date="2025-12-25")),
        example("Create birthday party event on 2025-05-18",     tc("calendar", action="create", date="2025-05-18", title="Birthday party")),
        example("What's on my schedule for March 10 2025?",      tc("calendar", action="list", date="2025-03-10")),
        example("Add reminder Doctor Checkup on 2025-11-05",     tc("calendar", action="create", date="2025-11-05", title="Doctor Checkup")),
        example("Create event Project Review on 2025-09-01",     tc("calendar", action="create", date="2025-09-01", title="Project Review")),
        example("Show calendar for 2025-02-14",                  tc("calendar", action="list", date="2025-02-14")),
        example("New event: Weekly Sync on 2025-10-07",          tc("calendar", action="create", date="2025-10-07", title="Weekly Sync")),
        example("What meetings are scheduled for 2025-07-04?",   tc("calendar", action="list", date="2025-07-04")),
        example("Add event Gym Session on 2025-04-12",           tc("calendar", action="create", date="2025-04-12", title="Gym Session")),
    ]


def convert_examples():
    return [
        example("Convert 100 miles to kilometers",       tc("convert", value=100,  from_unit="miles",       to_unit="kilometers")),
        example("How many kg is 180 pounds?",            tc("convert", value=180,  from_unit="pounds",      to_unit="kilograms")),
        example("Convert 72 Fahrenheit to Celsius",      tc("convert", value=72,   from_unit="Fahrenheit",  to_unit="Celsius")),
        example("Turn 5 gallons into liters",            tc("convert", value=5,    from_unit="gallons",     to_unit="liters")),
        example("How many feet in 10 meters?",           tc("convert", value=10,   from_unit="meters",      to_unit="feet")),
        example("500 grams in ounces",                   tc("convert", value=500,  from_unit="grams",       to_unit="ounces")),
        example("Convert 1 inch to cm",                  tc("convert", value=1,    from_unit="inches",      to_unit="centimeters")),
        example("2.5 liters to milliliters",             tc("convert", value=2.5,  from_unit="liters",      to_unit="milliliters")),
        example("100 C to F",                            tc("convert", value=100,  from_unit="Celsius",     to_unit="Fahrenheit")),
        example("100 F to C",                            tc("convert", value=100,  from_unit="Fahrenheit",  to_unit="Celsius")),
        example("What is 1 stone in kg?",                tc("convert", value=1,    from_unit="stone",       to_unit="kilograms")),
        example("200 miles to km",                       tc("convert", value=200,  from_unit="miles",       to_unit="kilometers")),
        example("3 gallons to liters",                   tc("convert", value=3,    from_unit="gallons",     to_unit="liters")),
        example("15 meters to feet",                     tc("convert", value=15,   from_unit="meters",      to_unit="feet")),
        example("300 grams to ounces",                   tc("convert", value=300,  from_unit="grams",       to_unit="ounces")),
        example("Convert 50 lbs to kilograms",           tc("convert", value=50,   from_unit="pounds",      to_unit="kilograms")),
        example("How many cm is 6 inches?",              tc("convert", value=6,    from_unit="inches",      to_unit="centimeters")),
        example("Convert 0 Celsius to Fahrenheit",       tc("convert", value=0,    from_unit="Celsius",     to_unit="Fahrenheit")),
    ]


def currency_examples():
    return [
        example("Convert 100 USD to EUR",               cur(100,  "USD", "EUR")),
        example("How much is 500 GBP in PKR?",          cur(500,  "GBP", "PKR")),
        example("Exchange 250 EUR to JPY",              cur(250,  "EUR", "JPY")),
        example("1000 PKR to USD",                      cur(1000, "PKR", "USD")),
        example("What's 75 AUD in CAD?",                cur(75,   "AUD", "CAD")),
        example("Convert 200 dollars to rupees",        cur(200,  "USD", "PKR")),
        example("2000 INR to GBP",                      cur(2000, "INR", "GBP")),
        example("50 CHF to USD",                        cur(50,   "CHF", "USD")),
        example("750 EUR to GBP",                       cur(750,  "EUR", "GBP")),
        example("500 USD to EUR",                       cur(500,  "USD", "EUR")),
        example("150 CAD to USD",                       cur(150,  "CAD", "USD")),
        example("1000 JPY to USD",                      cur(1000, "JPY", "USD")),
        example("300 AED to PKR",                       cur(300,  "AED", "PKR")),
        example("How much is 80 USD in GBP?",           cur(80,   "USD", "GBP")),
        example("Convert 450 EUR to CHF",               cur(450,  "EUR", "CHF")),
    ]


def sql_examples():
    return [
        example("Show all users from the users table",           tc("sql", query="SELECT * FROM users")),
        example("Count orders placed today",                     tc("sql", query="SELECT COUNT(*) FROM orders WHERE DATE(created_at) = CURRENT_DATE")),
        example("Get the top 10 products by sales",              tc("sql", query="SELECT * FROM products ORDER BY sales DESC LIMIT 10")),
        example("List all active subscriptions",                 tc("sql", query="SELECT * FROM subscriptions WHERE status = 'active'")),
        example("Find customers in Karachi",                     tc("sql", query="SELECT * FROM customers WHERE city = 'Karachi'")),
        example("Update user 42 email to test@example.com",      tc("sql", query="UPDATE users SET email = 'test@example.com' WHERE id = 42")),
        example("Delete all expired sessions",                   tc("sql", query="DELETE FROM sessions WHERE expires_at < NOW()")),
        example("Show all orders",                               tc("sql", query="SELECT * FROM orders")),
        example("Count number of active users",                  tc("sql", query="SELECT COUNT(*) FROM users WHERE status = 'active'")),
        example("List inactive subscriptions",                   tc("sql", query="SELECT * FROM subscriptions WHERE status = 'inactive'")),
        example("Find customers in Lahore",                      tc("sql", query="SELECT * FROM customers WHERE city = 'Lahore'")),
        example("Get products sorted by price descending",       tc("sql", query="SELECT * FROM products ORDER BY price DESC")),
        example("Show orders from last 7 days",                  tc("sql", query="SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL 7 DAY")),
        example("Find all pending payments",                     tc("sql", query="SELECT * FROM payments WHERE status = 'pending'")),
    ]


def multi_turn_examples():
    return [
        # Currency chain
        example(
            "Convert that to euros",
            cur(100, "USD", "EUR"),
            history=[
                {"role": "user",      "content": "How much is 100 USD in GBP?"},
                {"role": "assistant", "content": cur(100, "USD", "GBP")},
            ]
        ),
        example(
            "Now convert the same to GBP",
            cur(500, "EUR", "GBP"),
            history=[
                {"role": "user",      "content": "Convert 500 EUR to JPY"},
                {"role": "assistant", "content": cur(500, "EUR", "JPY")},
            ]
        ),
        example(
            "And that in JPY?",
            cur(200, "USD", "JPY"),
            history=[
                {"role": "user",      "content": "200 USD to EUR"},
                {"role": "assistant", "content": cur(200, "USD", "EUR")},
                {"role": "user",      "content": "and to GBP?"},
                {"role": "assistant", "content": cur(200, "USD", "GBP")},
            ]
        ),
        # Temperature cross-tool
        example(
            "What is that in Fahrenheit?",
            tc("convert", value=22, from_unit="Celsius", to_unit="Fahrenheit"),
            history=[
                {"role": "user",      "content": "Weather in Rome?"},
                {"role": "assistant", "content": tc("weather", location="Rome", unit="C")},
                {"role": "user",      "content": "It says 22 degrees"},
                {"role": "assistant", "content": "22°C — a pleasant day!"},
            ]
        ),
        # Calendar create → list
        example(
            "Show me what's on that day",
            tc("calendar", action="list", date="2025-09-15"),
            history=[
                {"role": "user",      "content": "Create event Review on 2025-09-15"},
                {"role": "assistant", "content": tc("calendar", action="create", date="2025-09-15", title="Review")},
            ]
        ),
        # 3-turn calendar
        example(
            "Add another event called Budget Meeting on that same day",
            tc("calendar", action="create", date="2025-06-10", title="Budget Meeting"),
            history=[
                {"role": "user",      "content": "List my events on 2025-06-10"},
                {"role": "assistant", "content": tc("calendar", action="list", date="2025-06-10")},
                {"role": "user",      "content": "Create Quarterly Review on 2025-06-10"},
                {"role": "assistant", "content": tc("calendar", action="create", date="2025-06-10", title="Quarterly Review")},
            ]
        ),
        # Ambiguous — no history, should refuse
        example(
            "Convert that to miles",
            "I don't have a value to convert. Could you tell me how many kilometers or other units you'd like to convert?"
        ),
        example(
            "What's the weather there?",
            "I'm not sure which city you're referring to. Could you specify a location?"
        ),
    ]


def refusal_examples():
    return [
        example("Tell me a joke",            "Ha! Why don't scientists trust atoms? Because they make up everything! I can help with weather, calendar, conversions, currency, and SQL queries too."),
        example("What is the meaning of life?", "That's a great philosophical question! I'm best suited for weather, calendar events, unit conversions, currency exchange, and SQL queries."),
        example("Book me a flight to Lahore",   "I don't have a flight booking tool. I can help with weather, calendar events, conversions, currency exchange, or SQL queries."),
        example("Send an email to my boss",     "I can't send emails — no email tool available. I can help with calendar scheduling, currency, conversions, weather, or database queries."),
        example("What's the stock price of Apple?", "I don't have a stock lookup tool. My tools are: weather, calendar, unit convert, currency exchange, and SQL."),
        example("Play music for me",            "I can't play music — no media tool available. Can I help with weather, a calendar event, or a quick conversion?"),
        example("Translate this to French: hello", "I don't have a translation tool. I can assist with weather, calendar, unit conversions, currency, and SQL."),
        example("Order pizza for me",           "I can't place food orders — no ordering tool available. My tools are: weather, calendar, convert, currency, sql."),
        example("Hello!",                       "Hi there! I'm Pocket-Agent. I can help with weather, calendar events, unit conversions, currency exchange, and SQL queries. What would you like to do?"),
        example("Who are you?",                 "I'm Pocket-Agent, your on-device assistant. I can handle weather lookups, calendar management, unit conversions, currency exchange, and SQL queries — all offline."),
        example("Set an alarm for 7am",         "I don't have an alarm tool. My available tools are: weather, calendar, convert, currency, and sql."),
        example("Remind me to call mum",        "I can create a calendar event for that! Just tell me the date and I'll schedule it. Otherwise, I don't have a reminder/alarm tool."),
        example("What time is it?",             "I don't have access to a real-time clock. For time-based scheduling, I can create calendar events — just let me know the date!"),
        example("Turn off the lights",          "I don't have smart home controls. My tools are limited to weather, calendar, conversions, currency, and SQL."),
        example("Call my wife",                 "I can't make phone calls — no telephony tool available. I can help with weather, calendar, conversions, currency exchange, or SQL."),
    ]


def adversarial_examples():
    return [
        # Typos
        example("wether in londoon?",             tc("weather", location="London", unit="C")),
        example("convrt 100 km to mils",          tc("convert", value=100, from_unit="kilometers", to_unit="miles")),
        example("curency 200 USD to EUR",         cur(200, "USD", "EUR")),
        example("Calender for march 15 2025",     tc("calendar", action="list", date="2025-03-15")),
        example("selekt all from users",          tc("sql", query="SELECT * FROM users")),
        example("weathre in paris?",              tc("weather", location="Paris", unit="C")),
        example("100 dolars to euro",             cur(100, "USD", "EUR")),
        example("curent exchange 50 CHF to USD",  cur(50, "CHF", "USD")),

        # Urdu/Hindi + English
        example("Mujhe Paris ka mausam batao Celsius mein",   tc("weather", location="Paris", unit="C")),
        example("100 dollar ko rupay mein convert karo",       cur(100, "USD", "PKR")),
        example("aaj ka weather London",                       tc("weather", location="London", unit="C")),
        example("aaj ka schedule dikhao 2025-04-01",          tc("calendar", action="list", date="2025-04-01")),
        example("500 rupaye USD mein",                         cur(500, "PKR", "USD")),
        example("Mujhe London ka mausam batao",                tc("weather", location="London", unit="C")),

        # Spanish + English
        example("¿Cual es el clima en Madrid hoy?",            tc("weather", location="Madrid", unit="C")),
        example("convertir 50 millas a kilometros",            tc("convert", value=50, from_unit="miles", to_unit="kilometers")),
        example("convertir 100 dolares a euros",               cur(100, "USD", "EUR")),

        # Arabic
        example("ما هو الطقس في دبي؟",                        tc("weather", location="Dubai", unit="C")),
        example("كم درجة الحرارة في لندن؟",                   tc("weather", location="London", unit="C")),

        # Unit abbreviations / alternate spellings
        example("50 lbs to kilos",                            tc("convert", value=50, from_unit="pounds", to_unit="kilograms")),
        example("convert 10 metre to foot",                   tc("convert", value=10, from_unit="meters", to_unit="feet")),
        example("100 C to F",                                 tc("convert", value=100, from_unit="Celsius", to_unit="Fahrenheit")),
        example("Convert 2 tons to kg",                       tc("convert", value=2, from_unit="metric tons", to_unit="kilograms")),

        # Hallucination bait → refusal
        example("Weather in Atlantis",                        "Atlantis is a mythical location. Please provide a real city name."),
        example("Weather in Wakanda",                         "Wakanda is a fictional country. Please provide a real city name."),
        example("Convert 100 Galactic Credits to USD",        "Galactic Credits is not a real ISO currency. Please use a valid currency code like USD, EUR, GBP."),
        example("List events in the Batcave on 2025-01-01",   tc("calendar", action="list", date="2025-01-01")),  # SQL on odd table is valid

        # SQL on unusual (but valid) table names
        example("SQL query on table unicorns",                tc("sql", query="SELECT * FROM unicorns")),
        example("Get everything from the dragons table",      tc("sql", query="SELECT * FROM dragons")),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def build_dataset(seed: int = 42) -> list:
    all_examples = (
        weather_examples()
        + calendar_examples()
        + convert_examples()
        + currency_examples()
        + sql_examples()
        + multi_turn_examples()
        + refusal_examples()
        + adversarial_examples()
    )
    random.seed(seed)
    random.shuffle(all_examples)
    return all_examples


def compute_sha256(examples: list) -> str:
    """Compute SHA-256 over all user prompts — used by grader to check overlap."""
    prompts = []
    for item in examples:
        for msg in item["messages"]:
            if msg["role"] == "user":
                prompts.append(msg["content"])
    combined = "\n".join(sorted(prompts))
    return hashlib.sha256(combined.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate Pocket-Agent training data")
    parser.add_argument("--out",  default="data/train.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42,        help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    examples = build_dataset(seed=args.seed)

    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    sha = compute_sha256(examples)
    print(f"Generated {len(examples)} training examples → {args.out}")
    print(f"SHA-256 of training prompts: {sha}")
    print("(Grader will compare this hash against public_test.jsonl — zero overlap verified.)")


if __name__ == "__main__":
    main()
