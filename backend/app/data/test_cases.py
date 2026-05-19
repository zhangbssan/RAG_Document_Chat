test_cases = [
    {
        "id": "test_001",
        "question": "What corporate vision and core values are described in the employee handbook?",
        "expected_answer": "The company aims to be a trusted technology solutions infrastructure provider guided by innovation with purpose, uncompromising integrity, radical collaboration, and excellence in execution",
        "expected_keywords": ["corporate vision", "technology solutions", "innovation", "integrity", "collaboration", "excellence"],
        "expected_document": "employee_handbook_en.pdf",
        "expected_page": 2
    },
    {
        "id": "test_002",
        "question": "What battery-saving profiles are described for extending SmartX Smartwatch operation?",
        "expected_answer": "The manual describes an Endurance Profile that disables Always-On Display, throttles heart-rate polling, and deactivates continuous blood oxygen monitoring for 14 to 18 days, plus an Extreme Ultra-Low Power Profile that limits functions to timekeeping, basic step tracking, and emergency call forwarding for up to 30 days",
        "expected_keywords": ["Endurance Profile", "Always-On Display", "heart-rate polling", "blood oxygen", "Extreme Ultra-Low Power", "30 days"],
        "expected_document": "product_manual_en.pdf",
        "expected_page": 4
    },
    {
        "id": "test_003",
        "question": "What service level, support, and pricing terms are defined for the cloud storage service?",
        "expected_answer": "The agreement provides a 5TB dedicated cloud storage instance, guarantees at least 99.9% monthly uptime excluding scheduled maintenance, offers 24/7/365 support with a critical ticket response window not exceeding 2 hours, and prices the premium tier at RMB 1,999 per fiscal year",
        "expected_keywords": ["5 Terabytes", "99.9%", "maintenance", "24/7/365", "2 hours", "RMB 1,999"],
        "expected_document": "service_agreement_en.pdf",
        "expected_page": 1
    },
    {
        "id": "test_004",
        "question": "What paid annual leave and sick leave rules are stated in the employee handbook?",
        "expected_answer": "After twelve months of continuous service, full-time employees receive 5 days of paid annual leave, increasing by one day per year up to 15 days, while paid sick leave requires a valid medical certificate within 48 hours of returning to duty",
        "expected_keywords": ["twelve months", "5 days", "annual leave", "15 days", "sick leave", "medical certificate"],
        "expected_document": "employee_handbook_en.pdf",
        "expected_page": 5
    },
    {
        "id": "test_005",
        "question": "What encryption, privacy, and liability protections are defined in the cloud storage agreement?",
        "expected_answer": "The agreement states that stored data uses zero-knowledge AES-256 encryption, Party A must not view or mine customer files except under a valid court order, and Party A's liability is capped at the amount paid for the active annual subscription cycle except for gross negligence or intentional criminal misconduct",
        "expected_keywords": ["AES-256", "zero-knowledge", "privacy", "court order", "liability cap", "annual subscription"],
        "expected_document": "service_agreement_en.pdf",
        "expected_page": 3
    }
]
