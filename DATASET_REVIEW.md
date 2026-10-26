# Public Dataset Feasibility Review

**Status:** Source review completed; no external dataset has been downloaded or used to generate new experimental results.

## Finding

I did not identify an openly accessible public dataset that provides claim-level ground-truth labels for the four specific mechanisms in the OW-ERF synthetic benchmark (upcoding, unbundling, phantom billing, and collusion) in the intended Indian health-insurance setting.

There are public U.S. Medicare sources that may support a **separate, provider-level external analysis**, but they are not a direct substitute for the current leave-one-mechanism-out experiment.

## Candidate sources

### 1. CMS Medicare Physician & Other Practitioners — by Provider

Official dataset: https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider

CMS describes this public-use dataset as an annual aggregation of Original Medicare Part B services, utilization, payments and submitted charges by provider (NPI). The latest release listed on the official page during this review is calendar year 2024.

**Potential use:** Provider-level billing-pattern profiling or a carefully labelled secondary anomaly-detection experiment.

**Limitations:** These data are aggregated at provider level, not the same claim-level longitudinal setting used by OW-ERF. They do not provide verified labels for the four fine-grained mechanisms in the paper. The U.S. Medicare system also differs from India's commercial health insurance and AB-PMJAY context.

### 2. CMS Medicare Physician & Other Practitioners — by Provider and Service

Official dataset series: https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners

CMS also provides a provider-and-service product organized by provider and HCPCS service code. The associated data dictionary is available at https://data.cms.gov/resources/medicare-physician-other-practitioners-by-provider-and-service-data-dictionary.

**Potential use:** A more detailed provider/service profile than provider totals alone.

**Limitations:** Provider/service aggregates are not ground-truth claim labels and do not identify which individual procedure was fraudulent or which novel mechanism caused the anomaly.

### 3. HHS Office of Inspector General — List of Excluded Individuals/Entities (LEIE)

Official download page: https://oig.hhs.gov/exclusions/leie-database-supplement-downloads/

The OIG provides a downloadable CSV of exclusions currently in effect and monthly updates. Exclusion is a provider/entity status, not a per-claim annotation of upcoding, unbundling, phantom billing or collusion. Exclusion reasons vary, so the precise reason and exclusion dates must be considered rather than treating every provider/service row as a proven fraudulent claim.

**Potential use:** Weak provider-level labels for a separate evaluation, with exclusion-time and reason-aware matching and clear caveats.

**Limitations:** LEIE labels cannot be interpreted as ground-truth per-claim fraud labels or as labels for the four OW-ERF mechanisms. Record linkage and date handling would need to be documented and independently validated.

### Supporting published research

Bauder and Khoshgoftaar, *The Detection of Medicare Fraud Using Machine Learning Methods with Excluded Provider Labels* (AAAI FLAIRS-31, 2018): https://cdn.aaai.org/ocs/17617/17617-77660-1-PB.pdf

This study combines public Medicare provider/service data with exclusion-based provider labels. It explicitly notes that the exclusion list lacks details linking specific procedures to fraud and that its labels therefore depend on assumptions at provider level.

A later data-centric study reports constructing Medicare fraud classification datasets from CMS data and OIG exclusion information: https://pmc.ncbi.nlm.nih.gov/articles/PMC10173919/

## Recommendation for this project

1. Keep the current synthetic leave-one-mechanism-out results labelled as **preliminary synthetic evidence**.
2. Do not represent the CMS/LEIE combination as a direct validation of unseen fraud-mechanism discovery.
3. If further external validation is required, treat it as a **separate provider-level experiment** with its own features, label definition, time-window handling, evaluation protocol and limitations. It would require changes to the current prototype and should not be claimed complete until it has actually been implemented and run.
4. If scope or time does not permit that separate experiment, retain the current synthetic benchmark, state the dataset limitation plainly and make the external validation a future-work item.

## References reviewed

- CMS, Medicare Physician & Other Practitioners — by Provider: official dataset documentation (link above).
- CMS, Medicare Physician & Other Practitioners — provider/service series and data dictionary (links above).
- HHS OIG, LEIE database and supplement downloads (link above).
- Bauder, R. A., and Khoshgoftaar, T. M. (2018), *The Detection of Medicare Fraud Using Machine Learning Methods with Excluded Provider Labels* (link above).
- *Data-Centric AI for Healthcare Fraud Detection* (link above).

This is a feasibility note, not a report of a new dataset experiment. No external results have been added to the paper or results table.
