# Random-number seed list for all manuscript figures

This file records the seeds of the numerical calculations that produced the current manuscript figures. DMP, spectral-root, and data aggregation results are deterministic once their input network or processed MetroFlow data have been fixed. A range such as `6001--6108` denotes the base seed for each configuration; the seven infection-probability scans inside one early-growth MC estimate use `base + 73*l`, with `l=0,...,6`.

| Manuscript figure | Random component | Seed(s) |
| --- | --- | --- |
| 1 | Schematic drawing | None |
| 2 | SIS-transition schematic | None |
| 3 | ER bilayer and spring layout | `42`; no MC |
| 4 | ER bilayer used for the capacity sweep | `1313`; no MC |
| 5 | ER bilayer; prevalence MC points | network `202`; MC bases `2000+q`, `q=0,...,9` |
| 6 | BA bilayer; trajectory MC points | network `303`; MC bases `3000, 3001, 3002` |
| 7 | ER/BA synthetic-validation ensemble | network bases `500,501,502` (ER) and `600,601,602` (BA); MC bases `6001--6108` |
| 8 | ER/WS/BA strict-grid realizations | `13000+o+41a+811b+s`, where `o=0,20000,40000` for ER, WS, BA; `a,b=0,...,24`; `s=0,...,9`. The second layer uses this seed plus `7919`. The exact realized values are in the archived `heterogeneous_boundary_instances.csv` `seed` column. No MC is plotted in the current Figure 8. |
| 9 | BA bilayer; one-way-coupling MC points | network `606`; MC bases `6100--6112` |
| 10 | Two BA bilayers; bidirectional-coupling MC points | networks `720` (large gap) and `721` (small gap); MC bases `7600--7612` and `8600--8612`, respectively |
| 11 | MetroFlow conceptual construction | None |
| 12 | Shanghai station map | None; produced deterministically from the released processed MetroFlow table |
| 13 | Hourly Perron-root variation | None; deterministic spectral calculation from the processed MetroFlow table |
| 14 | Five-window data-share DMP--MC scan | MetroFlow MC base `20260729`; baseline window `w` uses `20260729+100000w`; coupled point `(w,j)` uses `20260729+100000w+10000+(100+j)*1000`, with `w=0,...,4`, `j=0,...,20` |
| 15 | Three-allocation-rule DMP--MC comparison | MetroFlow MC base `20260729`; coupled point `(w,m,j)` uses `20260729+100000w+10000+(1000+100m+j)*1000`, with `m=0,1,2` and `j=0,...,4` |
| 16 | Five-$\omega$ DMP--MC sensitivity comparison | MetroFlow MC base `20260729`; coupled point `(w,o,j)` uses `20260729+100000w+10000+(2000+100o+j)*1000`, with `o=0,...,4` and `j=0,...,4` |

For MetroFlow Figures 14, 15, and 16, infection-probability scan `l=0,...,6` uses the configuration seed plus `10007*l`; the bootstrap generator uses the configuration seed plus `900001`. The retained MC settings file records the base seed `20260729`, and refined points use the same deterministic schedule with more realizations rather than selected replacement seeds.
