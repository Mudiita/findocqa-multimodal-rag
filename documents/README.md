# Source documents

The 30 PDFs are not included in this repository (they belong to the issuers).
Download each one from the issuer's own website and save it under `documents/<category>/` with the exact
file name below, so that it matches `dataset_manifest.csv`. Then build the index with `python main.py ingest`.

Folders: `documents/insurance/` (15), `documents/mutual_funds/` (10), `documents/fixed_deposits/` (5).

| doc_id | Provider | Product | File |
|---|---|---|---|
| INS01 | Aditya Birla Health | Activ Health | `documents/insurance/adityabirla_activhealth.pdf` |
| INS02 | Bajaj Allianz | My Health Care | `documents/insurance/bajajallianz_myhealthcare.pdf` |
| INS03 | Care Health | Care prospectus | `documents/insurance/carehealth_prospectus.pdf` |
| INS04 | Digit | Health Care Plus | `documents/insurance/digit_healthcareplus.pdf` |
| INS05 | Generali Central | Group Health | `documents/insurance/generalicentral_grouphealth.pdf` |
| INS06 | HDFC ERGO | Combined health | `documents/insurance/hdfcergo_combined.pdf` |
| INS07 | ICICI Lombard | Complete Health | `documents/insurance/icicilombard_complete.pdf` |
| INS08 | New India Assurance | Mediclaim | `documents/insurance/newindia_mediclaim.pdf` |
| INS09 | Niva Bupa | ReAssure/Rise | `documents/insurance/nivabupa_rise.pdf` |
| INS10 | Reliance General | Health Global | `documents/insurance/reliancegeneral_healthglobal.pdf` |
| INS11 | SBI General | Group Health | `documents/insurance/sbigeneral_grouphealth.pdf` |
| INS12 | Star Health | Basic plan | `documents/insurance/starhealth_basic.pdf` |
| INS13 | Star Health | Premium plan | `documents/insurance/starhealth_premium.pdf` |
| INS14 | Tata AIG | MediCare | `documents/insurance/tataaig_medicare.pdf` |
| INS15 | Universal Sompo | Complete Healthcare | `documents/insurance/universalsompo_completehealthcare.pdf` |
| MF01 | SBI | Nifty Next 50 Index | `documents/mutual_funds/sbi_niftynext50_index.pdf` |
| MF02 | Tata | Nifty 50 Index | `documents/mutual_funds/tata_nifty50_index.pdf` |
| MF03 | HDFC | Large Cap | `documents/mutual_funds/hdfc_largecap.pdf` |
| MF04 | Axis | Bluechip | `documents/mutual_funds/axis_bluechip_largecap.pdf` |
| MF05 | SBI | Magnum Midcap | `documents/mutual_funds/sbi_magnum_midcap.pdf` |
| MF06 | HDFC | Mid-Cap Opportunities | `documents/mutual_funds/hdfc_midcap.pdf` |
| MF07 | HDFC | Corporate Bond | `documents/mutual_funds/hdfc_corporatebond_debt.pdf` |
| MF08 | ICICI Prudential | Short Term | `documents/mutual_funds/icici_shortterm_debt.pdf` |
| MF09 | ICICI Prudential | Balanced Advantage | `documents/mutual_funds/icici_balancedadvantage_hybrid.pdf` |
| MF10 | HDFC | Hybrid Equity | `documents/mutual_funds/hdfc_hybridequity_hybrid.pdf` |
| FD01 | SBI | Retail term deposit | `documents/fixed_deposits/sbi_fd.pdf` |
| FD02 | HDFC Bank | Retail FD | `documents/fixed_deposits/hdfc_fd.pdf` |
| FD03 | ICICI Bank | Retail FD | `documents/fixed_deposits/icici_fd.pdf` |
| FD04 | Kotak Mahindra Bank | Retail FD | `documents/fixed_deposits/kotak_fd.pdf` |
| FD05 | Bank of Baroda | Retail FD | `documents/fixed_deposits/bob_fd.pdf` |

Rates and factsheets change over time; the dates the evaluation used are in the `doc_date` column of the manifest
where known. A different version of a document will give different answers from the graded results.
