> **Last updated: 2026-11-27 16:15 (Berlin)** — W16 added (why not DBSCAN / HDBSCAN); W15 corrected: the LOF k actually used is 40 (edge step of D15, notebook 03 in repo commit 424b74d), not 50; archived notes removed from the project (see Related files)

# Thesis writing snippets — ready sentences, their reasoning, and where they go

**What this file is:** sentences proposed in chat for the thesis text, each with (1) the sentence, (2) the reasoning behind it, as we argued it, (3) where it goes in the thesis, (4) what to check before using it. Grouped by chapter. One entry = one W-number (W1, W2, …), so it can be referred to in chat. Longer drafts also carry a **Persian version** (نسخهٔ فارسی) of the same text.

**Related files:**
- `claude/defence_notes.md` — questions an examiner may ask, with checked answers (Q1, Q2, …). Snippets here are for *writing*; defence notes are for *answering*. Some topics appear in both.
- `Thesis_Execution_Plan.xlsx` — Decisions (D#) and Limitations (L#) are the reference; this file only links to them.
- Archived notes (deleted from the project on 27 Nov 2026; history only, kept in the GitHub repo): `results/feature_engineering_plan.md`, `results/sentence_construction_notes.md`, `docs/archive/notebook02_review_changes.md` (C# / N# / X# items of the notebook-02 review; C cell numbers refer to the notebook before the rewrite).

**Status labels:** `draft` (sentence proposed, not yet in the thesis) · `check numbers` (a number in the sentence must be re-checked against the current notebook run) · `used` (copied into the thesis; note the section).

---

## Chapter 3 — Methodology (data, feature engineering)

### W1. Entry-level fields are encoded once per entry (D13) — `draft`

**Sentence:**
> "In the synthetic ledger, the user, the dates and the tax rate are identical on all lines of an entry (0 exceptions in 102,429 entries, notebook 02, cell_02_03). The user is therefore encoded once per entry instead of once per line slot, so that the same fact is not counted up to four times in every distance. For the user and the dates this matches real ERP systems, where they are document header fields; the tax rate, however, is a line-item attribute in real ledgers (e.g. SAP BSEG-MWSKZ) and would have to be encoded per line slot there."

**Reasoning (فارسی):**
- چک داده (cell_02_03) نشان داد این فیلدها در هیچ سندی بیش از یک مقدار ندارند.
- تکرار یک فیلد ثابت در ۴ slot همان واقعیت را تا ۴ بار در فاصله‌ها (LOF، autoencoder) می‌شمارد و وزنش را بی‌دلیل بالا می‌برد.
- تفاوت مهم: `user` و تاریخ‌ها در SAP واقعاً در header سند (BKPF) هستند، پس این تصمیم در دنیای واقعی هم درست است. `tax_rate` در SAP سطح خط است (BSEG-MWSKZ)، پس این ساده‌سازی فقط به خاطر این دیتاست است (L1).
- `tax_rate` و `n_lines` در تنظیم اصلی خاموش‌اند، چون از الگوی حساب‌ها تعیین می‌شوند (چک افزونگی ۲۶ نوامبر، بخش ۶.۱ نوت‌بوک ۰۲).
- یک دقت: عدد ۰ فقط نشان می‌دهد «در این دیتاست ثابت است»؛ اینکه در ledger واقعی هم ثابت باشد را عدد نشان نمی‌دهد — آن از دانش SAP می‌آید.

**Where:** Ch. 3, feature engineering → entry-level vs line-level fields. Limitation L1 in Ch. 5.

**Check before use:** the code guard is in the notebook (cell_02_04), not yet inside `src/encode.py` `entry_table`. If you write "the code stops with an error on a ledger with mixed tax rates", first move the check into `entry_table` (proposed in chat, 26 Nov).

---

### W2. Terminology: debit/credit sequence vs booking pattern — `draft` (terminology rule, not one sentence)

**Rule for the whole thesis:**
- **debit/credit sequence** = the D/C letters of the lines in stored (CSV) order, e.g. `DDC`; 5 values in the ledger (cell_02_05, variable `csv_pattern`).
- **booking pattern** = account + side of every line after D9 sorting, e.g. `D1576 D3400 C1600`; 29 values (`src/figures.py`, `pattern`; used in D16 sentence construction and in section 6.1).

**Sentence (if a definition is needed):**
> "We call the set of accounts and debit/credit sides of an entry, listed in the fixed line order of Section X, its booking pattern; the ledger contains 29 distinct booking patterns."

**Reasoning (فارسی):** کلمهٔ «pattern» در نوت‌بوک ۰۲ دو معنی متفاوت دارد (۵ مقدار در برابر ۲۹ مقدار). اگر در متن هر دو را «pattern» بنامی، خواننده عددها را قاطی می‌کند. مثال: CCD و DCC دو «sequence» متفاوت‌اند ولی هر دو یک بدهکار و دو بستانکار دارند؛ بعد از D9 هر دو با بدهکار شروع می‌شوند.

**Where:** Ch. 3, first place where patterns are mentioned (data description or D9). Optional: rename `pattern` → `cd_sequence` in cell_02_05 and cell_02_08.

---

### W3. Descriptive tables on the whole ledger vs the FY1-only label rule (D6) — `draft`

**Sentence:**
> "Tables that only describe the ledger (e.g. the number of entries and anomalies per debit/credit sequence) use both fiscal years; every figure or table that supports a feature decision uses labels of FY2023 only, so that no label of the reporting year enters a design choice."

**Reasoning (فارسی):** cell_02_05 برچسب هر دو سال را جمع می‌زند، در حالی که خود نوت‌بوک (بخش ۶) قاعده گذاشته که فقط FY1 برچسب ببیند (D6). این leakage نیست چون هیچ تصمیمی از آن جدول گرفته نمی‌شود، ولی ممتحن دقیق ممکن است ناهماهنگی را ببیند. همچنین: تفاوت نرخ ناهنجاری بین sequenceها (DC 1.06٪، DCC 0.51٪) را تفسیر نکن — از اینجا می‌آید که ژنراتور کدام نوع M را در کدام تراکنش گذاشته.

**Where:** Ch. 3, experimental setup (next to D6).

---

### W6. Amount scaling: the robust step before min-max has no effect (D10) — `draft`

**Sentence:**
> "Amounts and the entry lag are transformed with log1p and then min-max scaled to [0, 1] within each fiscal year (D10). Gronewald et al. (2024) apply a robust scaling (median, interquartile range) between these two steps; since min-max scaling removes any preceding shift and positive rescaling, that step does not change the resulting values, and the influence of extreme amounts is reduced by the logarithm alone."

**Reasoning (فارسی):**
- کد (`Scaler` در `src/encode.py`) سه مرحله دارد: `log1p` ← robust `(x − median) / IQR` ← min-max.
- min-max هر تبدیل خطیِ قبل از خودش را خنثی می‌کند: `minmax(a·x + b) = minmax(x)` برای a > 0. robust دقیقاً همین تبدیل است. پس خروجی نهایی = `minmax(log1p(x))`.
- چک عددی (۲۶ نوامبر): بیشترین اختلاف بین دو مسیر 2.2e-16 — فقط خطای گرد کردن.
- در متن ننویس «RobustScaler اثر داده‌های پرت را کم می‌کند» — در این زنجیره این کار را فقط log1p می‌کند.

**Where:** Ch. 3, numeric features / scaling (with D10; merge with W7).

**Check before use:** check in the Gronewald et al. (2024) paper that their pipeline is really robust → min-max in this order before citing it this way.

---

### W7. Numeric features: one transformation, justified field by field (D10) — `draft`, `check numbers`

**Which numeric fields exist (code: `build_onehot`, `src/encode.py`):**

| Field | In main setting? | Transformation | Range |
|---|---|---|---|
| `amount` (one value per line slot, PAD = 0) | yes | log1p → min-max per fiscal year (fitted on all lines of the year, all slots pooled) | [0, 1] |
| `lag_days` (entered date − document date, days) | yes | log1p → min-max per fiscal year | [0, 1] |
| `nwh` | yes | none (already 0/1) | {0, 1} |
| `weekend` | yes | one-hot weekday / Sat / Sun (categorical, not scaled) | {0, 1} |
| `n_lines` | off (D13) | fixed formula (n − 2) / 2 | {0, 0.5, 1} |
| `promptly` | off (D4) | fixed formula (p − 1) / 2 | {0, 0.5, 1} |
| `nwh_dist` | off (D11) | divided by the year maximum (zero for most entries, bounded → no log needed) | [0, 1] |
| hour / weekday sin, cos | off (D11) | none | [−1, 1] |

So in the main setting **only two fields are scaled: `amount` and `lag_days`**, and they share one transformation.

**Data behind the choice (computed 26 Nov from journal_entries_4203.csv):**

| | amount (lines) FY2023 / FY2024 | lag_days (entries) FY2023 / FY2024 |
|---|---|---|
| min | 0.06 / 0.10 | 0 / 0 (zeros: 2.8 % / 3.1 %) |
| median | 99.53 / 103.49 | 5 / 5 |
| 99th percentile | 3,334 / 3,463 | 26 / 16 |
| max | 22,500 / 22,500 (rent, normal) | 181 / 64 (normal entries) |
| skewness raw → after log1p | 10.3 → −0.16 / 9.2 → −0.17 | 11.6 → −0.46 / 6.1 → −0.77 |

**Draft text (English):**
> "Two numeric features are used in the main setting: the amount of each line slot and the entry lag in days. Both are non-negative and strongly right-skewed (skewness 10.3 for amounts and 11.6 for the lag in FY2023): most amounts are below EUR 310 while monthly rent reaches EUR 22,500, and most entries are recorded within a week while a few are recorded months later. Without a transformation, these few large values would decide the scale and all typical values would be pressed close to zero. Both features are therefore transformed with log1p, which reduces the skewness to about −0.2 (amount) and −0.5 (lag); log1p rather than log is used because 2.8 % of the entries have a lag of zero days. The result is min-max scaled to [0, 1] within each fiscal year (D10), so that the numeric features have the same range as the one-hot columns and neither block dominates the distances only through its units. The same transformation is used for both fields because they share the properties that motivate it — non-negative values, a long right tail and no meaningful negative or upper bound; it was not chosen by comparing results."

**Points to mention as properties / limitations (not yet decided to change):**
1. *The amount scale is shared by all slots and all accounts* — a tax line of EUR 20 and a rent line of EUR 22,500 are on one scale. An amount relative to its account (e.g. z-score of log amount per account) would show "large for this account"; noted earlier as an idea for ablation / Future Work.
2. *Min-max per year depends on one value per year.* The FY2023 lag maximum is 181 days (one normal entry), the FY2024 maximum is 64 days, so the same delay of 30 days becomes 0.66 in FY2023 and 0.82 in FY2024. This is consistent inside each year (detectors are fitted per year, D6), but one extreme entry sets the scale of the whole year. Alternative if needed: use a fixed cap or a high percentile (e.g. 99th) as the upper bound and let larger values exceed 1.
3. The robust step is redundant (W6).

**نسخهٔ فارسی:**
> «در تنظیم اصلی دو ویژگی عددی استفاده می‌شود: مبلغ هر جای خط (slot) و فاصلهٔ روز بین تاریخ سند و تاریخ ثبت (lag). هر دو نامنفی و به‌شدت چوله به راست‌اند (چولگی ۱۰٫۳ برای مبلغ و ۱۱٫۶ برای lag در FY2023): بیشتر مبلغ‌ها زیر ۳۱۰ یورو است ولی اجارهٔ ماهانه به ۲۲٬۵۰۰ یورو می‌رسد، و بیشتر سندها ظرف یک هفته ثبت می‌شوند ولی چندتایی ماه‌ها بعد. بدون تبدیل، همین چند مقدار بزرگ مقیاس را تعیین می‌کنند و همهٔ مقادیر معمولی نزدیک صفر فشرده می‌شوند. بنابراین هر دو با log1p تبدیل می‌شوند که چولگی را به حدود ۰٫۲− (مبلغ) و ۰٫۵− (lag) می‌رساند؛ log1p به جای log به این دلیل است که ۲٫۸٪ سندها lag صفر دارند. نتیجه در هر سال مالی جداگانه با min-max به بازهٔ [0, 1] برده می‌شود (D10)، تا ویژگی‌های عددی هم‌بازهٔ ستون‌های one-hot باشند و هیچ بلوکی فقط به خاطر واحدش بر فاصله‌ها غالب نشود. برای هر دو فیلد یک تبدیل به کار رفته چون ویژگی‌هایی که این تبدیل را لازم می‌کنند مشترک‌اند — مقادیر نامنفی، دُم بلند به راست و نبودِ حد بالای معنادار؛ این انتخاب با مقایسهٔ نتایج انجام نشده است.»

**Reasoning (فارسی) — پاسخ به سؤال «آیا نباید فیلد به فیلد scaler مناسب می‌آوردم؟»:**
- بله، **توجیه** باید فیلد به فیلد باشد. ولی نتیجهٔ توجیه می‌تواند «یک تبدیل برای هر دو» باشد، به شرطی که با عدد نشان بدهی هر دو فیلد همان ویژگی‌ها را دارند (جدول بالا). ایراد ممتحن به «یک روش برای همه» نیست؛ به «یک روش بدون دلیل برای هر فیلد» است.
- `nwh_dist` نتیجهٔ دیگری می‌گیرد (بیشتر صفر، کران‌دار → log لازم نیست) — یعنی بررسی فیلد به فیلد واقعاً به نتیجه‌های متفاوت می‌رسد.

**Where:** Ch. 3, feature engineering → numeric features and scaling. Table of numbers can go in the appendix if space is short.

**Check before use:** recompute the numbers in the notebook (they were computed outside it on 26 Nov) — new cells N1/N2 in `notebook02_review_changes.md`.

---

### W8. New users, accounts or tax rates in a new year (categories, D6, D10, L9) — `draft`

**Facts (26 Nov):**
- Category lists (accounts, users, tax rates) are built from the whole ledger in one pass, so in this thesis no value can be "unseen". The error in `_onehot` ("value outside the category list") is a guard for applying a stored category list to other data; the code has no such step today (`build_onehot` always rebuilds).
- The dataset already contains a case: **account 2020 "Prior-period expenses" appears only in FY2024, on one line (entry p_2)**. Users (6) and tax rates (2) are the same in both years.
- Detectors (D6) and scalers (D10) are fitted per fiscal year. So the category list can also be per year without any mismatch: the columns only have to agree within the year the detector is fitted on.
- Dimension on a real ledger: users add U columns once; accounts add (A + 1) × number of slots — e.g. 500 accounts and 4 slots = 2,004 columns, 10 slots = 5,010. The detectors still run, but distances in very sparse high-dimensional data lose contrast (LOF, kNN) and Isolation Forest wastes splits on near-constant columns. Remedies: account grouping by chart class, frequency encoding, embedding (L9, RQ2).

**Draft text (English):**
> "All fitted parts of the encoding — the category lists, the scalers and the detectors — are built from the fiscal year that is analysed (D6, D10). A user, account or tax rate that appears for the first time in that year therefore does not break the encoding; it simply becomes a column of that year. The ledger contains one such case: account 2020 (prior-period expenses) is used only once, in FY2024. What a new category does change is the score: a value that occurs in very few entries makes these entries rare, so they are ranked high. For an auditor this can be wanted — a new user posting entries or a newly used account is a common reason for review — but it can also produce false positives, for example when a new employee joins (L9). If a model fitted on one year were applied to the next year, values not seen in the fitting year would have to be mapped to a shared 'other' category; encoding them as all zeros would itself create a unique and therefore isolated pattern. A sentence embedding can place an unseen account near accounts with similar names, which is one argument for the semantic encoding (RQ2)."

**نسخهٔ فارسی:**
> «همهٔ بخش‌های fitشدهٔ encoding — فهرست دسته‌ها، scalerها و detectorها — از همان سال مالی که تحلیل می‌شود ساخته می‌شوند (D6، D10). بنابراین کاربر، حساب یا نرخ مالیاتی که اولین بار در آن سال ظاهر شود encoding را نمی‌شکند؛ فقط یک ستونِ همان سال می‌شود. این دفتر یک نمونه از این حالت دارد: حساب 2020 (هزینه‌های مربوط به دورهٔ قبل) فقط یک بار، در FY2024، به کار رفته است. چیزی که یک دستهٔ جدید عوض می‌کند امتیاز است: مقداری که در سندهای خیلی کمی آمده آن سندها را نادر می‌کند و رتبهٔ بالا می‌گیرند. برای حسابرس این می‌تواند مطلوب باشد — ثبت سند توسط کاربر جدید یا استفاده از حسابی تازه دلیل رایجی برای بررسی است — ولی می‌تواند false positive هم بسازد، مثلاً وقتی کارمند جدیدی می‌آید (L9). اگر مدلی که روی یک سال fit شده روی سال بعد اجرا شود، مقادیری که در سال fit دیده نشده‌اند باید به یک دستهٔ مشترک "سایر" نگاشت شوند؛ کدکردن آن‌ها به صورت همه‌صفر خودش یک الگوی یکتا و در نتیجه منزوی می‌سازد. یک sentence embedding می‌تواند حساب دیده‌نشده را بر اساس اسمش کنار حساب‌های هم‌معنا بگذارد، که یکی از دلایل encoding معنایی است (RQ2).»

**Reasoning (فارسی):**
- نگرانی «اگر در FY2 مقدار جدیدی بیاید خطا می‌دهیم» در این پایان‌نامه رخ نمی‌دهد، چون فهرست دسته‌ها یک‌جا از کل دفتر ساخته می‌شود. خطا فقط محافظی است برای حالتی که فهرست ذخیره‌شده روی دادهٔ دیگری اعمال شود.
- «حساس نبودن به کاربر/حساب جدید» دو معنی دارد: (۱) کد نشکند — با fit سالانه برآورده است؛ (۲) امتیاز تحت تأثیر قرار نگیرد — این نه ممکن است و نه لزوماً مطلوب، چون detector بدون نظارت دقیقاً چیز نادر را بالا می‌برد. در متن هر دو معنی را جدا کن.
- مسئلهٔ ابعاد در شرکت واقعی: کاربر مسئلهٔ اصلی نیست (۲۰ کاربر = ۱۴ ستون بیشتر)؛ حساب است، چون در هر slot تکرار می‌شود.

**Where:** Ch. 3 (encoding: category lists, per-year fitting) — first three sentences; Ch. 5 Discussion / L9 (transfer, new and batch users, dimension) — the rest.

**Optional code change:** build the category lists per fiscal year inside the D10 loop (as the scalers), to match the text exactly; today they come from the whole ledger, which is harmless here (an unused column is all zeros in that year) but is not "fitted on the analysed year".

---

### W9. Line order (D9): what the 50 % order changes mean — and what they do not mean — `draft`

**Facts (cells 02_08, 02_09; checked again 26 Nov):**
- 51,253 of 102,429 entries (50.0 %) are stored in the CSV in a different order than the D9 order (all DCC and CCD entries, 33,192 of 33,312 DDC, the DDDC entry; no DC entry).
- **But each of the 29 booking patterns is stored in exactly one order** (cell_02_09). So in this ledger, two entries with the same content would get the same vector even without sorting. The 50 % does **not** show that the problem "same content → different vector" occurs here. (The archived plan of 21 Nov said "the problem is real in this dataset" — that conclusion was too strong; cell_02_09 of 25 Nov corrected it.)
- What D9 does change here is the **alignment of slots across patterns**: in CSV order, 60 entries have a credit line in slot 1, while all other entries start with a debit. All 60 are one booking pattern — C3400 Cost of merchandise, C1576 Input tax, D1600 Trade payables, text "Reversal …" (e.g. r_1 "Reversal Hedge Trimmer") — i.e. **reversals of purchases**, all labelled normal. After D9 every entry starts with a debit, so slot 1 means "a debit line" in every entry.
- D9 is not free: example purchase `a_1` vs the reversal `r_1`. CSV order: a_1 = D3400, D1576, C1600 and r_1 = C3400, C1576, D1600 — same accounts in the same slots, only sides flipped (squared distance in the account+side block = 6, d ≈ 2.45). D9 order: a_1 = D1576, D3400, C1600 and r_1 = D1600, C1576, C3400 — the accounts no longer line up (squared distance 8, d ≈ 2.83). Sorting by side first separates a booking from its reversal a little more. An order-free encoding (aggregate per account × side, "bag of lines") would avoid any such choice → Future Work (already listed under D9 option 5).

**Draft text (English):**
> "Before padding, the lines of each entry are sorted by debit/credit side, account number and amount (D9), so that each slot has the same meaning in every entry and two entries with the same content always receive the same vector. In the CSV, half of the entries (51,253 of 102,429) are stored in a different order. In this ledger, however, the stored order is consistent within each of the 29 booking patterns, so the sorting does not repair inconsistent orders here; it guarantees the property for extracts that give no such guarantee, and it aligns the slots across patterns — for example, the 60 reversals of purchases are stored credit-first and would otherwise be the only entries with a credit line in the first slot. The rule is a choice, not a neutral step: it places a booking and its reversal slightly further apart than the stored order does, because the accounts no longer fall into the same slots. An order-free encoding of the lines is left for future work."

**نسخهٔ فارسی:**
> «پیش از padding، خط‌های هر سند بر اساس طرف بدهکار/بستانکار، شمارهٔ حساب و مبلغ مرتب می‌شوند (D9)، تا هر جای خط در همهٔ سندها یک معنا داشته باشد و دو سند با محتوای یکسان همیشه بردار یکسان بگیرند. در فایل CSV نیمی از سندها (۵۱٬۲۵۳ از ۱۰۲٬۴۲۹) به ترتیب دیگری ذخیره شده‌اند. اما در این دفتر ترتیب ذخیره درون هر یک از ۲۹ الگوی ثبت ثابت است، پس مرتب‌سازی اینجا ترتیب‌های ناسازگار را تعمیر نمی‌کند؛ این ویژگی را برای خروجی‌هایی تضمین می‌کند که چنین تضمینی ندارند، و جای خط‌ها را بین الگوها هم‌تراز می‌کند — مثلاً ۶۰ ثبت برگشتیِ خرید (reversal) با خط بستانکار ذخیره شده‌اند و وگرنه تنها سندهایی بودند که در جای اول خط بستانکار دارند. این قاعده یک انتخاب است، نه یک قدم خنثی: یک ثبت و ثبت برگشتی‌اش را کمی دورتر از ترتیب ذخیره‌شده قرار می‌دهد، چون حساب‌ها دیگر در جای‌های یکسان نمی‌افتند. encoding بدون ترتیب برای خط‌ها به کار آینده واگذار می‌شود.»

**Reasoning (فارسی):**
- عدد ۵۰٪ چشمگیر است ولی به تنهایی چیزی ثابت نمی‌کند؛ سؤال درست این است که آیا *محتوای یکسان* با *ترتیب متفاوت* ذخیره شده — و جواب در این دفتر «نه» است (۰ از ۲۹).
- پس دلیل D9 اینجا دو چیز است: (۱) تضمین برای داده‌ای که ژنراتور ندارد (خروجی واقعی SAP)، (۲) هم‌ترازی slotها بین الگوها.
- اینکه بگوییم D9 هزینه هم دارد (مثال خرید و ثبت برگشتی‌اش) متن را قوی‌تر می‌کند: نشان می‌دهد تصمیم آگاهانه بوده، نه پیش‌فرض.

**Where:** Ch. 3, encoding → line order (D9). The purchase/reversal example can go in a footnote or in Ch. 5 next to L2 / Future Work.

**Check before use:** distances 2.45 / 2.83 are for the account + side block only (amount columns ignored); state that if the numbers are given.

---

### W10. "Amount of an entry" = sum of its debit lines (figures, J1 validation) — `draft` (code change agreed 26 Nov)

**Where the code uses one amount per entry (26 Nov):**
- **Not in the model.** The encoding keeps one amount per line slot (4 amount columns); no entry-level amount is a feature.
- `src/figures.py` `entry_frame`: `amount = max(line amounts)` — used only for the figures (e.g. item text vs price, D1).
- `src/load.py` `build_entries`: `max_amount = max(line amounts)` — used by the own J1 implementation in `src/jets.py` (validation only; J1 in `meta` follows the dataset's `top_n`).

**Decision (Mohsen, 26 Nov):** change both to the **sum of the debit lines** (standard definition); rename `max_amount` → `entry_amount`.

**Check (26 Nov, whole ledger):** max line amount = sum of debit lines in **100 %** of the 102,429 entries; sum of debits = sum of credits in 100 % (every entry is balanced). So no number changes.

**Why max works here — and why it is not the right general definition:**
- In double-entry bookkeeping the size of an entry is the sum of its debits (= sum of its credits).
- The largest line equals that sum only when one side has a single line. That holds for every pattern in this ledger (DC, DDC, DCC, CCD, DDDC — always one line alone on one side).
- In a real ledger (e.g. a payroll entry with several debits and several credits) the largest line is smaller than the entry total, so "max" would understate the size of the entry.

**Draft text (English):**
> "Where one amount per entry is needed (descriptive figures and the validation of the large-amount test J1), the entry amount is the sum of its debit lines, which equals the sum of its credit lines in a balanced entry. In this ledger every entry has a single line on one side, so this sum also equals the largest line of the entry (checked for all 102,429 entries)."

**نسخهٔ فارسی:**
> «هر جا یک مبلغ برای کل سند لازم است (نمودارهای توصیفی و اعتبارسنجی آزمون مبلغ بزرگ J1)، مبلغ سند برابر جمع خط‌های بدهکار آن است، که در سند متوازن با جمع خط‌های بستانکار برابر است. در این دفتر هر سند در یک طرف فقط یک خط دارد، پس این جمع با بزرگ‌ترین خط سند هم برابر است (برای هر ۱۰۲٬۴۲۹ سند چک شده).»

**Reasoning (فارسی):** «max» در این دیتاست درست جواب می‌دهد ولی دلیلش یک ویژگی دیتاست است (یک خط تنها در یک طرف)، نه یک اصل حسابداری. تعریف «جمع بدهکار» اصل حسابداری است و روی هر دفتر متوازن درست است؛ پس در متن تعریف اصولی را بنویس و برابری با max را به‌عنوان چک بیاور.

**Where:** Ch. 3, data description (one sentence) or a footnote at the first figure that shows an entry amount.

**Check before use:** how the dataset defines `top_n` (per line or per entry) — see the Data in Brief paper, Table 3; if the dataset uses the largest line, say that J1 follows the dataset definition.

---

### W11. Free text `text` (D1): reasons already recorded + the "item × amount" check — `draft`

**Where the reasons are:** Decisions sheet, row D1 (reviewed with Claude on 25 Nov; Persian column filled; thesis sentence drafted; status "To confirm", rejection risk low), FE-DataChecks rows of 21 Nov (325 values, Lawn Mower 3.0 % from price) and 26 Nov (figure `fe_d1_text_vs_price` recomputed on FY1). Short version of the reasons: (1) dataset limitation — 325 values, no anomaly type manipulates the text, Lawn Mower's higher rate comes from price, no manual / automatic flag so the real audit signal cannot be tested; (2) RQ2 fairness; (3) same as Gronewald et al. 2024 / 2026; (4) real texts are templates + free text → Future Work (L5); (5) near-unique monthly texts would look novel in an embedding.

**Gap found on 26 Nov (chat "D1 review") and closed the same day:** the one argument against reason (1) was: *a cheap item booked at an expensive item's price (e.g. a Rake at a Lawn Mower price) can only be seen if the model knows the item — i.e. through `text`*. Check (26 Nov, whole ledger): 845 of the 850 anomalies have one of the 10 item texts. For each item, the range (min–max) of the entry amounts of its normal entries was taken; **0 of the 845 anomalies have an amount outside the normal range of their own item.** So no anomaly in this ledger is of the "wrong price for this item" kind, and leaving `text` out loses nothing on this data. On a real ledger this combination could matter → Future Work (L5).

**Sentence to add after the D1 sentence (English):**
> "One case where the text could add information is an amount that is unusual for the item but not for the ledger, such as a cheap item booked at the price of an expensive one. In this ledger no anomaly is of that kind: every anomalous entry with an item text has an amount within the range of the normal entries of the same item."

**نسخهٔ فارسی:**
> «تنها حالتی که متن می‌تواند اطلاعات اضافه بدهد، مبلغی است که برای آن کالا غیرعادی است ولی برای کل دفتر نه — مثلاً کالای ارزانی که با قیمت یک کالای گران ثبت شده باشد. در این دفتر هیچ ناهنجاری‌ای از این نوع نیست: مبلغ هر سند ناهنجاری که متن کالا دارد، در بازهٔ مبلغ سندهای عادیِ همان کالا قرار دارد.»

**Where:** Ch. 3, feature engineering → D1 paragraph. Add a row to FE-DataChecks (26 Nov, "item × amount": FY2023: 0 of 420 anomalies with an item text lie outside their item's normal amount range — notebook 02 §7.1; whole ledger 0 of 845; D1, L5).

**Check before use:** recompute in the notebook next to figure 6.2 (new small cell) so the number comes from the notebook. The range test uses min–max of normal entries (whole ledger, labels used only to split normal / anomalous, descriptive) — a strict test; a looser test (e.g. 1st–99th percentile) could flag a few anomalies and should be reported if used. Figure 6.2 title: use "The expensive items carry the higher anomaly rates" (the relation is not monotonic: Gardening Gloves, the cheapest item, 0.60 % > mid-priced items 0.29–0.48 %).

---

### W12. Redundancy check: what the functional dependencies and Cramér's V can and cannot show (D4, D13) — `draft`

**Facts (cell_02_14, whole ledger, no labels):**
- Functional dependencies found: booking pattern → `n_lines`; booking pattern → `tax_rate`; `lag_days` → `promptly`. Also hold (not in the notebook list yet): hour → `nwh`, weekday → `weekend`.
- The booking pattern stands for the account + side block of the slots (one-to-one after D9).
- Pairwise Cramér's V is low for `user` vs pattern (0.14), yet 21 of the 29 patterns — the rare ones, 638 entries together — are posted by a single user only (e.g. the M1 pattern D1576 D3400 C1000 only by Max), while the five large patterns are posted by all six users (corrected 26 Nov: an earlier note said 7 of 29). `weekend` vs pattern / `n_lines` = 0.30 because ~30 % of 2-line entries (payments, receipts) are posted on weekends and almost no 3-line entries — generator structure.

**Draft text (English):**
> "Every fact should enter the distances once. Entry-level fields were therefore checked for functional dependencies on the whole ledger, without labels: a field B is dropped when every value of a field A determines exactly one value of B. Three such dependencies were found: the booking pattern determines the number of lines and the tax rate, and the delay in days determines the dataset's delay class `promptly`. In each case the dependent field is dropped, because the determining field carries all of its information and more. The pairwise association between fields (Cramér's V) is shown only to document redundancy; it is not used to judge whether a field is useful. A low pairwise value does not mean that a field carries no signal: the user is only weakly associated with the booking pattern (V = 0.14), yet 21 of the 29 booking patterns are posted by a single user, and the injected anomalies of types M1–M6 lie in such user–pattern combinations, which a pairwise measure cannot show."

**نسخهٔ فارسی:**
> «هر واقعیت باید فقط یک بار در فاصله‌ها شمرده شود. به همین دلیل فیلدهای سطح سند روی کل دفتر و بدون برچسب از نظر وابستگی تابعی بررسی شدند: فیلد B کنار گذاشته می‌شود وقتی هر مقدار فیلد A دقیقاً یک مقدار B را تعیین کند. سه وابستگی پیدا شد: الگوی ثبت تعداد خط‌ها و نرخ مالیات را تعیین می‌کند، و تأخیر بر حسب روز کلاس تأخیر دیتاست (`promptly`) را. در هر مورد فیلد وابسته کنار گذاشته می‌شود، چون فیلد تعیین‌کننده همهٔ اطلاعات آن و بیشتر را دارد. وابستگی جفتی بین فیلدها (Cramér's V) فقط برای مستندکردن افزونگی نشان داده می‌شود و برای قضاوت دربارهٔ مفید بودن یک فیلد به کار نمی‌رود. مقدار جفتی پایین به معنی نداشتن سیگنال نیست: کاربر فقط وابستگی ضعیفی با الگوی ثبت دارد (V = 0.14)، ولی ۲۱ الگو از ۲۹ الگو فقط توسط یک کاربر ثبت شده‌اند، و ناهنجاری‌های تزریق‌شدهٔ نوع M1 تا M6 دقیقاً در چنین ترکیب‌هایی از کاربر و الگو هستند، که یک معیار جفتی نمی‌تواند نشان دهد.»

**Extra evidence (26 Nov, cell_02_18 numbers):** dropping the three dependent fields does not change how many entries can be told apart: the main setting has 24,207 distinct vectors among the 51,064 FY2023 entries both with them (135 columns, v2) and without them (131 columns, v3). Optional sentence: *"Removing the three dependent fields leaves the number of distinct entry vectors unchanged (24,207 in FY2023), which confirms that they carried no additional information."*

**Reasoning (فارسی):** اگر heatmap را «ارزش فیلد» بخوانی، ممتحن می‌پرسد چرا `user` را با V = 0.14 نگه داشتی. جواب: heatmap فقط برای افزونگی است؛ ارزش با ablation سنجیده می‌شود. اینکه چرا فیلد وابسته حذف می‌شود (نه تعیین‌کننده) را هم صریح بنویس.

**Where:** Ch. 3, feature engineering → redundancy check (next to D4, D13). Heatmap figure in Ch. 3 or appendix.

**Check before use:** "M1–M6 lie in user–pattern combinations" — confirm per type from the Data in Brief generator description (week 1 notes: Max for M1 / M5, Fred for M2 / M3 / M6, David for M4); M3–M6 are also timing / amount anomalies, so say "partly" if needed.

---

### W13. Delay figure: the steps at 10 and 30 days come from the generator (D4, D6) — `draft`

**Facts (cell_02_16, FY2023 only):** anomaly rate by delay: 0–4 days 0.69 %, 5–9 0.48 %, 10–14 2.51 %, 15–19 4.88 % (82 entries), 20–29 1.85 % (162), 30–44 18.4 % (196), 45–59 9.8 %, 60–89 9.4 % (32), 90+ 12.5 % (56); base rate 0.83 %. Of the FY2023 anomalies with a delay of 30+ days, **60 of 62 are type M4**; of those with 10–29 days, **60 of 95 are type M3** (26 Nov).

**Reading:** the risk does not rise smoothly with the delay; it jumps exactly at the `promptly` edges (10 and 30 days) because the generator injected M3 and M4 with these delay classes. The Decisions sheet (D4) already says this.

**Draft text (English):**
> "In FY2023 the anomaly rate jumps at delays of 10 and 30 days (from below 1 % to 2.5 % and to 18 %). These are the edges of the dataset's delay classes, and almost all anomalies above them belong to the two injected delay types (M3: 60 of 95 anomalies with 10–29 days; M4: 60 of 62 with 30 days or more). The steps therefore reflect how the anomalies were generated, not a property of real ledgers. The model receives the delay in days, which contains the class edges exactly, but not the class itself."

**نسخهٔ فارسی:**
> «در FY2023 نرخ ناهنجاری در تأخیر ۱۰ و ۳۰ روز جهش می‌کند (از کمتر از ۱٪ به ۲٫۵٪ و به ۱۸٪). این‌ها مرزهای کلاس‌های تأخیر خود دیتاست‌اند، و تقریباً همهٔ ناهنجاری‌های بالای آن‌ها از دو نوع تأخیری تزریق‌شده هستند (M3: ۶۰ از ۹۵ ناهنجاری با ۱۰ تا ۲۹ روز؛ M4: ۶۰ از ۶۲ با ۳۰ روز یا بیشتر). پس این پله‌ها نشان‌دهندهٔ نحوهٔ تولید ناهنجاری‌هاست، نه ویژگی دفترهای واقعی. مدل تأخیر را بر حسب روز دریافت می‌کند، که مرزهای کلاس را دقیقاً در خود دارد، ولی خود کلاس را نه.»

**Reasoning (فارسی):**
- عنوان فعلی نمودار («Risk rises with the delay, in steps at 10 and 30 days») درست است ولی نمی‌گوید پله‌ها از کجا می‌آیند؛ ممتحن ممکن است آن را «یافته» بخواند.
- نرخ یکنواخت نیست (0–4 روز 0.69٪ > 5–9 روز 0.48٪؛ 15–19 روز 4.9٪ > 20–29 روز 1.9٪)، و چند بازه فقط ۳۰ تا ۸۰ سند دارند → بازهٔ اطمینان (مثلاً Wilson) روی میله‌ها بگذار.
- ربط به D4: اینکه `promptly` خاموش است به خاطر وابستگی تابعی است (W12)، نه به خاطر «دلبخواهی بودن مرزها» — در این دیتاست مرزها دلبخواهی نیستند، خود ژنراتور با آن‌ها ناهنجاری ساخته. پس در متن D4 دلیل «arbitrary edges» را ضعیف‌تر بگو یا حذف کن.

**Where:** Ch. 3, feature engineering → time features (D4), with the figure; Ch. 5 Limitations (L8: synthetic labels).

**Check before use:** M3 / M4 definitions in the Data in Brief paper (Table 3); recompute the M-type counts in the notebook (new small cell after the figure).

---

### W14. Does the continuous delay hurt when the generator used the delay classes? (D4) — `draft`, `check numbers`

**Question (Mohsen, 26 Nov):** the generator injected M3 / M4 by the `promptly` classes; could our continuous `lag_days` then help less than `promptly`, or even hurt?

**Experiment (26 Nov, FY2023 only — allowed for choosing the main setting, D6; outside the notebook):** encoding as in notebook 02; IF 300 trees / 2,048 samples, AP = mean of 5 seeds; LOF k = 50 (or the largest duplicate group + 1 when larger); recall of M3 / M4 within the top 1,000 (seed 0).

| Variant | IF AP | LOF AP | LOF k | LOF R@1000 M3 | LOF R@1000 M4 |
|---|---|---|---|---|---|
| A main: `lag_days` | 0.057 | **0.116** | 50 | 0.00 | **0.87** |
| B `promptly` instead of `lag_days` | 0.068 | 0.046 | 171 (largest duplicate group 170) | 0.00 | 0.23 |
| C main + `promptly` | 0.066 | 0.101 | 50 | 0.23 | 0.00 |

**Reading:**
- The fear is not supported: with LOF the continuous delay gives the best AP and finds 87 % of M4 (30+ days) in the top 1,000; the far tail of the delay is rare in either form.
- Replacing the delay by the 3 classes makes many more entries identical (largest group 36 → 170), so LOF needs k = 171 and gets much worse — less resolution, more ties.
- M3 (10–29 days, mostly 10–14 days next to the dense 5–9-day mass) is hard for every variant; adding `promptly` lets LOF find 23 % of M3 but loses M4 and lowers AP — it shifts what is found, it does not add.
- IF differences are small (0.057–0.068) and IF finds few M3 / M4 in any variant.
- The class information is not lost from the pipeline: J3 (`promptly` ≥ 2) is one of the JET flags, so it enters H1, H2 and the combined score.

**Draft text (English):**
> "Because the generator injected the delay anomalies by the dataset's delay classes, using the delay in days instead of the class could in principle lose signal. A check on FY2023 shows the opposite for the main detector: with LOF, the delay in days gives a higher average precision than the delay class (0.116 vs 0.046) and places 87 % of the long-delay anomalies (M4) in the top 1,000, while the coarse class makes many entries identical and forces a much larger neighbourhood. Adding the class next to the delay shifts which delay type is found rather than improving the ranking (AP 0.101). The class itself still enters the method through the delayed-entry test J3."

**نسخهٔ فارسی:**
> «چون ژنراتور ناهنجاری‌های تأخیری را بر اساس کلاس‌های تأخیر دیتاست تزریق کرده، استفاده از تأخیر بر حسب روز به جای کلاس در اصل می‌توانست سیگنال را از دست بدهد. بررسی روی FY2023 برای detector اصلی خلاف این را نشان می‌دهد: با LOF، تأخیر بر حسب روز average precision بالاتری از کلاس تأخیر می‌دهد (0.116 در برابر 0.046) و ۸۷٪ ناهنجاری‌های تأخیر طولانی (M4) را در ۱۰۰۰ سند اول قرار می‌دهد، در حالی که کلاس درشت بسیاری از سندها را یکسان می‌کند و همسایگی بسیار بزرگ‌تری را تحمیل می‌کند. اضافه‌کردن کلاس در کنار تأخیر فقط عوض می‌کند کدام نوع تأخیر پیدا شود، و رتبه‌بندی را بهتر نمی‌کند (AP 0.101). خود کلاس همچنان از طریق آزمون ثبت با تأخیر (J3) وارد روش می‌شود.»

**Where:** Ch. 3 (D4, short) or Ch. 4 (ablation table); defence question "why not the generator's own classes?".

**Check before use:** rerun inside a notebook (03 or the ablation notebook) with the final code and record the rows in `results/runs.csv`; the numbers above are from a one-off run on 26 Nov. B differs from A in two ways (feature and k) — say so, or also run A with k = 171 for a clean comparison. Note: this one-off run used LOF k = 50; the main setting now uses k = 40 (W15).

---

### W15. Identical vectors and the LOF neighbourhood size (D15, L10) — `draft` (corrected 27 Nov: k used = 40)

**Facts (26 Nov, no labels, per fiscal year; cell_02_24 in repo commit 424b74d):**

| Feature setting | Columns | FY2023: distinct vectors / entries | entries sharing their vector | largest group | FY2024 largest group |
|---|---|---|---|---|---|
| main (v3) | 131 | 24,207 / 51,064 | 36,137 (71 %) | 36 | 36 |
| without time features (proposal variant) | 126 | 3,547 / 51,064 | 50,738 (99 %) | 187 | 174 |
| main + hour_cyc | 133 | 50,837 / 51,064 | 283 | 12 | 12 |
| main + wday_cyc | 133 | 34,358 / 51,064 | 25,794 | 18 | 18 |

- The largest group in the main setting is recurring payroll (e.g. monthly wage-tax payments with the same amount, user and time flags).
- The hour on a circle (`hour_cyc`) makes almost every entry unique; `nwh_dist` changes almost nothing (0 for 99 % of the entries). Notebook 02 §8 (cell_02_24) prints the table for every setting of `fe.ABLATIONS`. The duplicates in the main setting exist because time enters only as flags.

**k actually used (notebook 03, cells 03_05 – 03_07, repo commit 424b74d):** grid 50 / 100 / 200 on FY2023 → best k = 50 (AP 0.1162); it lies on the grid edge, so the D15 edge rule (`detectors.edge_extension`: k − 10, only if still larger than the largest duplicate group) tested k = 40 → AP 0.1179, which is used on FY2024 (AP 0.144, p_2 rank 212). Guard in code: `detectors.run` stops with an error if `n_neighbors <= max_duplicate_group(X)`. D15 is decided in Thesis_Execution_Plan.xlsx (Decisions sheet; rejection risk by supervisors very low).

**Draft text (English):**
> "After encoding, many entries are identical: in the main setting, 36,137 of the 51,064 FY2023 entries (71 %) share their vector with at least one other entry, and the largest group of identical entries has 36 members (also 36 in FY2024), mostly recurring payroll postings. For LOF, a point with k or more identical neighbours has a k-distance of zero, which makes its density estimate unstable (Campos et al., 2016). The neighbourhood size must therefore be larger than the largest group of identical vectors (k ≥ 37 here). The grid 50, 100, 200 was evaluated on FY2023; since the best value lay on the edge of the grid, one further step, k = 40, was evaluated as fixed in advance, and it gave the highest average precision (0.118) and is used. The constraint depends on the feature set: without time features the largest group has 187 members, so k ≥ 188 is required."

**نسخهٔ فارسی:**
> «بعد از encoding، بسیاری از سندها یکسان‌اند: در تنظیم اصلی، ۳۶٬۱۳۷ سند از ۵۱٬۰۶۴ سند FY2023 (۷۱٪) بردارشان با دست‌کم یک سند دیگر یکی است، و بزرگ‌ترین گروه سندهای یکسان ۳۶ عضو دارد (در FY2024 هم ۳۶)، که بیشترشان ثبت‌های تکراری حقوق و دستمزد هستند. در LOF، نقطه‌ای که k همسایهٔ یکسان یا بیشتر داشته باشد k-distance صفر دارد و برآورد چگالی‌اش ناپایدار می‌شود (Campos et al., 2016). بنابراین اندازهٔ همسایگی باید بزرگ‌تر از بزرگ‌ترین گروه بردارهای یکسان باشد (اینجا k ≥ 37). شبکهٔ ۵۰، ۱۰۰ و ۲۰۰ روی FY2023 ارزیابی شد؛ چون بهترین مقدار روی لبهٔ شبکه بود، طبق قاعده‌ای که از قبل تعیین شده بود یک قدم دیگر، k = 40، هم ارزیابی شد که بالاترین average precision را داد (0.118) و همان استفاده می‌شود. این قید به مجموعهٔ ویژگی‌ها بستگی دارد: بدون ویژگی‌های زمانی بزرگ‌ترین گروه ۱۸۷ عضو دارد و k ≥ 188 لازم است.»

**Reasoning (فارسی):**
- خروجی قدیمی سلول «groups of size > 1: 9280» تعداد **گروه‌ها** بود، نه سندها؛ عددی که برای متن معنی دارد تعداد سندهای دارای همزاد (۷۱٪) است. FY2 هم نشان داده می‌شود (بدون برچسب است، پس مجاز است) چون LOF روی FY2 هم اجرا می‌شود.
- AP با کم شدن k بالا رفته (۲۰۰ → ۱۰۰ → ۵۰ → ۴۰). سؤال محتمل داور: «چرا ۳۷ را تست نکردی؟» جواب: قاعدهٔ لبه از قبل تعیین شده بود (یک قدم آن طرف لبه)، و ادامهٔ جست‌وجو روی برچسب‌های FY2023 تنظیم را به supervised نزدیک می‌کرد.

**Where:** Ch. 3, detectors / hyperparameters (D15); Ch. 5 Limitations (L10: real ledgers repeat bookings too, groups can be much larger).

---

### W16. Why not DBSCAN or HDBSCAN (detector choice) — `draft` (27 Nov)

**Question (Mohsen, 27 Nov):** why does the thesis not use DBSCAN or HDBSCAN? Gronewald et al. (2024) used HDBSCAN, and the thesis builds on their H1 / H2, so an examiner is likely to ask why HDBSCAN was replaced by LOF.

**Draft text (English):**
> "Gronewald et al. (2024) used HDBSCAN as one of their three detectors. It is not used here. This thesis evaluates rankings under a review budget, which requires a score for every entry. DBSCAN does not provide one by design: it assigns each point to a cluster or to noise, and all noise points are tied. HDBSCAN provides a score, GLOSH (Campello et al., 2015), but GLOSH compares the density of a point with the highest density in its cluster, which is the same idea of relative local density as LOF. Since one detector per family is used (isolation, density, reconstruction), GLOSH would duplicate the density family. It would also face the same constraint from identical vectors as LOF (Section X), and it has more parameters to tune; in Gronewald et al. (2024) it was the weakest detector, which the authors attribute to its dependence on its configuration."

**نسخهٔ فارسی:**
> «Gronewald و همکاران (2024) از HDBSCAN به‌عنوان یکی از سه detector خود استفاده کردند. این روش در اینجا استفاده نمی‌شود. این پایان‌نامه رتبه‌بندی را زیر بودجهٔ بررسی ارزیابی می‌کند، که برای هر سند یک امتیاز لازم دارد. DBSCAN طبق طراحی‌اش چنین امتیازی نمی‌دهد: هر نقطه را به یک خوشه یا به نویز نسبت می‌دهد و همهٔ نقطه‌های نویز با هم مساوی‌اند. HDBSCAN امتیاز دارد، یعنی GLOSH (Campello et al., 2015)، ولی GLOSH چگالی یک نقطه را با بیشترین چگالیِ خوشه‌اش مقایسه می‌کند، که همان ایدهٔ چگالی نسبی محلیِ LOF است. چون از هر خانواده یک detector استفاده می‌شود (isolation، density، reconstruction)، GLOSH خانوادهٔ density را تکرار می‌کرد. همچنین با همان قیدِ بردارهای یکسانِ LOF روبه‌رو می‌شد (بخش X) و پارامترهای بیشتری برای تنظیم دارد؛ در Gronewald و همکاران (2024) ضعیف‌ترین detector بود، که نویسندگان علتش را وابستگی آن به پیکربندی دانسته‌اند.»

**Reasoning — how we argued it (27 Nov):**
1. First version used "the scikit-learn HDBSCAN has no outlier score" as a reason. **Mohsen rejected it:** a supervisor would answer "then use the `hdbscan` package". Correct — that is a tool argument, not a method argument; removed.
2. **DBSCAN:** the lack of a score is a property of the algorithm itself (Ester et al., 1996: core / border / noise), so it holds for every implementation. A hand-made score (e.g. distance to the nearest core point) would be a new, own method and is close to the kNN distance anyway.
3. **HDBSCAN:** has a score (GLOSH, `hdbscan` package), so the reasons must be different: same density family as LOF; same duplicate constraint (core distance 0 when a point has ≥ min_samples identical copies → min_samples ≥ 37, like D15); three main parameters (min_cluster_size, min_samples, cluster_selection_method) to tune on FY2023 labels, against the small-grid rule of D15; weakest detector in Gronewald et al. (2024), about 51–53 %, attributed by the authors to default settings and configuration dependence.
4. **Comparability:** neither the nine detectors of Gronewald, Scherer & Mehdiyev (2026 preprint) nor the 14 unsupervised methods of ADBench (Han et al., 2022) include DBSCAN or HDBSCAN (ADBench's cluster-based method is CBLOF).
5. Herreros-Martínez et al. (2025) used DBSCAN only univariately (column by column, eps = 1), not as a detector on the whole entry vector.
6. Honest fallback if an examiner insists: GLOSH could be run as one extra row in the ablation; it was left out because of scope, not because of a technical barrier.

**Where:** Ch. 3, detector selection (one paragraph); `defence_notes.md` (question "why not HDBSCAN like Gronewald et al. 2024?").

**Check before use:** read the GLOSH definition in Campello et al. (2015, ACM TKDD, "Hierarchical density estimates for data clustering, visualization, and outlier detection") — the wording above was written from memory; add Ester et al. (1996) for DBSCAN to the references. Rejection risk by supervisors: low (proposal approved with IF, LOF, AE).

---

## Chapter 3 / 4 — wording rules

### W4. Unsupervised detectors rank; they do not "predict normal" — `draft` (wording rule)

**Rule:** write "ranks high / low", "assigns a high anomaly score", "places in the top k" — not "predicts normal / anomalous" or "classifies". Also: between categorical fields say "association (Cramér's V)" or "functional dependency", and towards the label "anomaly rate per category / bin" — not "correlation".

**Sentence (if needed in Ch. 3):**
> "The detectors are fitted without labels; each returns an anomaly score per entry, and entries are ranked by this score. Labels are used only to choose hyperparameters on FY2023 and to evaluate the ranking on FY2024."

**Reasoning (فارسی):** سؤال ۲۶ نوامبر: «آیا مدل DDDCهای FY2 را عادی پیش‌بینی می‌کند چون تنها نمونه‌اش عادی است؟» — نه. IF / LOF / AE برچسب نمی‌بینند؛ چیزی را بالا می‌برند که **نادر** است، نه چیزی که برچسبش ۱ بوده. اگر سندهای ۴ خطی زیاد بودند، خوشهٔ خودشان را می‌ساختند و امتیازشان پایین می‌آمد — به خاطر **فراوانی**، نه برچسب. «یادگرفتن از یک نمونهٔ عادی» فقط در مدل supervised (مثل LightGBM مرجع در preprint گرونوالد) یا semi-supervised پیش می‌آید. همین‌طور: روشن/خاموش کردن ویژگی بر اساس رابطه با برچسب روش را عملاً نیمه‌نظارتی می‌کند — تصمیم‌ها فقط از بررسی‌های بدون برچسب، نمای برچسب‌دار فقط توصیفی و روی FY1.

**Where:** Ch. 3, detectors / evaluation; also useful in the defence.

---

## Chapter 5 — Discussion / Limitations

### W5. The single four-line entry p_2: rare in shape and in content — `draft`, `check numbers` (corrected 26 Nov, 17:15)

**Correction (26 Nov):** the first version called p_2 "a normal purchase invoice with two expense lines". That was wrong — the account names had not been read. The accounts are: 1700 *Other liabilities* (D 1,490.90), 2020 *Prior-period expenses* (D 63.36), 1576 *Deductible input tax, 19 %* (D 295.31), 1600 *Trade payables* (C 1,849.57); `ref_id` = p_1. This is the **settlement of a liability booked in the previous period** (an accrual is cleared when the invoice arrives; the difference goes to prior-period expenses). Account 2020 occurs **only once in the whole ledger** (this line) and account 1700 only twice. So p_2 is rare not only because it alone fills slot 4, but also in its content.

**Sentence (new version):**
> "The only four-line entry in the ledger (p_2, label normal) settles a liability from the previous period: it debits 'Other liabilities' and 'Prior-period expenses' together with input tax against trade payables. It is ranked 212th by LOF (FY2024, main setting v3). Its rarity has two sources: it alone fills the fourth line slot of the encoding, and it uses an account (prior-period expenses) that occurs nowhere else in the ledger. The second source is audit-relevant — prior-period items are a usual subject of cut-off testing — so an auditor would reasonably review this entry even though it is labelled normal. The first source is an artefact of the fixed-slot encoding; on real ledgers with many multi-line documents it would create false positives without any audit meaning (L2)."

**Reasoning — how we argued it (26 Nov):**
1. **Mohsen's first idea:** "It may be legitimate, but it is rare, so it gets a high score — and with a limited budget I would review it too."
2. **First answer (Claude):** partly right; but the entry is rare only in vector shape, not in accounting content, so "I would review it too" is weak and excuses every false positive.
3. **Correction after reading the account names:** the entry is also rare in content (unique account, prior-period booking), and prior-period items are audit-relevant (cut-off). So Mohsen's argument is **stronger than the first answer said** for this entry.
4. **What still holds:** (a) in the evaluation its label is 0, so in the top k it still counts as a false positive — an argument does not change the metric; (b) the slot-4 effect is a real encoding artefact and is the part that generalises to real ledgers (L2); (c) keep "statistical rarity" and "audit-relevant unusualness" apart: here both are present, for different reasons.
5. **Side fact:** the entry is in FY2024; FY2023 has no 4-line entry and no account 2020, so the hyperparameter choice on FY2023 (D15) never saw either effect.

**نسخهٔ فارسی (جمله):**
> «تنها سند چهارخطی دفتر (p_2، برچسب عادی) تسویهٔ یک بدهی از دورهٔ قبل است: «سایر بدهی‌ها» و «هزینه‌های دورهٔ قبل» را همراه با مالیات قابل کسر بدهکار و حساب پرداختنی‌های تجاری را بستانکار می‌کند. LOF آن را در رتبهٔ ۲۱۲ قرار می‌دهد (FY2024، تنظیم اصلی v3). نادر بودنش دو منبع دارد: تنها سندی است که جای خط چهارم encoding را پر می‌کند، و از حسابی (هزینه‌های دورهٔ قبل) استفاده می‌کند که در هیچ جای دیگر دفتر نیامده است. منبع دوم از نظر حسابرسی معنادار است — اقلام مربوط به دورهٔ قبل موضوع رایج آزمون cut-off هستند — پس حسابرس به‌طور منطقی این سند را بررسی می‌کند، هرچند برچسب آن عادی است. منبع اول محصول encoding با تعداد ثابت جای خط است؛ در دفترهای واقعی با سندهای چندخطی فراوان، false positive بدون معنای حسابرسی می‌سازد (L2).»

**Where:** Ch. 5, Discussion (false positives / encoding effects) and Limitations (L2). One sentence in Ch. 4 results if the rank is shown in a table.

**Check before use:**
- Ranks re-checked on 26 Nov with encoding v3 (131 columns, per-year scalers; notebook 03 run on a copy): LOF 212 (was 202), IF 1,918 (was 1,970) of 51,365 — still inside the top 250 for LOF. Use the number of the final notebook-03 run.
- Read p_1 (the `ref_id`) to confirm it is the accrual being settled before writing "settles a liability from the previous period".
- "Prior-period items are a usual subject of cut-off testing" — add a source (audit standard or textbook) or soften to "can be of interest for cut-off testing".

---

## Later snippets

(add here — W17, …)
