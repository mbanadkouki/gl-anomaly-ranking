Questions for the dataset authors (Gronewald et al.)

Dataset: Gronewald, Stephan, Rombach & Fettke (2026), Generating synthetic journal entries for audit analytics: method and dataset, Data in Brief 69, 113257. File: journal_entries_4203.csv (seed 42).

Plan: collect questions here and send one short email by 15 Dec 2026 (as questions, not corrections). Write the email in simple English.

Before sending: the generator notebook synthetic_data_4203.ipynb is published with the dataset on Zenodo. Read the code that sets top_n and the M1 injection first — it may answer Q1 and Q2 without an email.

Q1. How are ties at the 1,000th amount handled in top_n? (19 Nov 2026)

Observation

Entry amount = largest line amount; fiscal year = year of posting_date.
top_n flags exactly 1,000 entries per fiscal year (4,987 lines, always all lines of an entry).
Many entries share the 1,000th amount (amounts are quantity × fixed unit price):
FY	1,000th amount (EUR)	entries above	entries at the amount	taken by top_n
2023	2,667.50	967	122	33
2024	2,976.49	977	143	23
With ties included (RANK ≤ 1,000), J1 has 2,209 entries instead of 2,000 (209 more). All top_n entries are inside this set.
Checked (19 Nov): the tied entries in top_n cannot be reproduced by a simple order. Sorting the ties by file row order, posting_id, entered_date, document_date, user, number of lines or total amount gives 80–92 mismatches each. Ranking single lines instead of entries is far off (~770 mismatches). So the choice looks random or comes from an earlier step of the generator.

Question (draft) In each fiscal year many entries have exactly the 1,000th amount. How did you choose which of them get top_n = 1? Was it random, or was top_n computed at an earlier step of the generation (for example before the anomaly injection)?

فارسی: دیتاست از tieهای روی آستانه فقط تعدادی را برداشته؛ با هیچ ترتیب سادهٔ (ترتیب فایل، posting_id، تاریخ‌ها، user) بازتولید نمی‌شود، پس احتمالاً تصادفی یا در مرحلهٔ دیگری از ژنراتور ساخته شده. اول کد notebook ژنراتور را در Zenodo ببین. تصمیم من: در pipeline اصلی از top_n دیتاست استفاده می‌کنم؛ پیاده‌سازی خودم فقط برای validation است.

Q2. Why are 2 of the 120 M1 anomalies not in top_n? (19 Nov 2026)

Observation

The paper says M1 anomalies were selected from the 1,000 highest-amount candidates.
118 of 120 M1 entries have top_n = 1. The two others:
posting_id	FY	amount (EUR)	rank (ties = same rank)	top_n	high_cash
a_2744	2023	2,667.50	968	0	1
aa_774	2024	2,976.49	978	0	1
Both sit exactly at the 1,000th amount, so this is linked to Q1: the tie-break dropped them.
Effect on my work: none on the JET union (both are also caught by J2, union still catches 798 of 850), only J1 alone recall is 118/120.

Question (draft) Two M1 entries (a_2744, aa_774) have the 1,000th amount but top_n = 0. Was the candidate list for the M1 injection built in a different way from the final top_n flag?

فارسی: طبق مقاله همهٔ M1ها باید داخل top 1000 باشند، ولی ۲ تا بیرون افتاده‌اند. هر دو دقیقاً روی آستانه‌اند، پس احتمالاً به همان tie-break سؤال ۱ برمی‌گردد.

Later questions

(add here)