import re

# Sample text
text = """
Amjed Usman bought a laptop for $999.99 on 24/10/2025. 
He contacted support via email amjadusman98@gmail.com or phone 7994609113.
His friend John also paid $5.50 on 12/08/2024.
You can reach Sarah at sarah.johnson@company.co.uk.
"""

# 1️⃣ Find all names starting with a capital letter
names = re.findall(r'\b[A-Z][a-z]*\b', text)

# 2️⃣ Extract all email addresses
emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)

# 3️⃣ Find all 10-digit phone numbers
phones = re.findall(r'\b\d{10}\b', text)

# 4️⃣ Extract all dates in DD/MM/YYYY format
dates = re.findall(r'\d{2}/\d{2}/\d{4}', text)

# 5️⃣ Find all currency amounts (like $5.50, $3.25)
currencies = re.findall(r'\$\d+\.\d{2}', text)

# 🧾 Print all results
print("Names:", names)
print("Emails:", emails)
print("Phone Numbers:", phones)
print("Dates:", dates)
print("Currency Amounts:", currencies)
