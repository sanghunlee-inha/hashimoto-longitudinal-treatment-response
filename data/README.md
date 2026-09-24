# Data directory

This directory intentionally contains **no patient-level clinical data**.

Do not commit:

- EHR/EMR exports;
- patient identifiers;
- visit dates/timestamps;
- laboratory records tied to individuals;
- medication records tied to individuals;
- exact transition tables containing patient-level observations; or
- any intermediate file that could permit re-identification.

The analysis scripts should reference authorized local data outside the Git repository.
