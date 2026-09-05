# Tier A -> Tier B evaluation-base effect (the Glass-7 comparison caveat)

Glass-7 is scored on Tier B (Glass-7-capable files: 640 / 917.48 h / 177 seizures); every other config on Tier A (668 / 945.49 h / 181 seizures). This table re-scores each Tier A config's locked operating points on the Tier B file subset and reports the shift, so a Glass-7 vs (e.g.) Best-7 comparison can be read with the denominator change accounted for. Model = rf. Post-processing = locked default (merge_gap=8 s, min_event_duration=1 s). Only chb13/15/16/17/18/19 carry any non-Glass-7 files, so only their per-subject rows move.


## SZCORE

| config | scheme | criterion | micro-sens A->B | Δ | micro-FA/day A->B | Δ% | pass-rate A->B | Δ pp |
|---|---|---|---|---|---|---|---|---|
| Best-2 | 5fold | medication_titration | 0.230->0.235 | +0.005 | 9.093->9.345 | +2.8% | 17.4%->17.4% | +0.0 |
| Best-2 | 5fold | realtime_alert | 0.153->0.156 | +0.003 | 6.401->6.596 | +3.1% | 4.3%->4.3% | +0.0 |
| Best-4 | 5fold | medication_titration | 0.208->0.212 | +0.005 | 5.994->6.125 | +2.2% | 17.4%->17.4% | +0.0 |
| Best-4 | 5fold | realtime_alert | 0.169->0.173 | +0.004 | 3.658->3.769 | +3.1% | 13.0%->13.0% | +0.0 |
| Best-7 | 5fold | medication_titration | 0.224->0.229 | +0.005 | 4.521->4.659 | +3.1% | 17.4%->17.4% | +0.0 |
| Best-7 | 5fold | realtime_alert | 0.197->0.201 | +0.004 | 3.658->3.769 | +3.1% | 17.4%->17.4% | +0.0 |
| Full-18 | 5fold | medication_titration | 0.235->0.240 | +0.005 | 4.140->4.214 | +1.8% | 26.1%->26.1% | +0.0 |
| Full-18 | 5fold | realtime_alert | 0.169->0.173 | +0.004 | 1.753->1.806 | +3.1% | 17.4%->17.4% | +0.0 |
| Glass-2 | 5fold | medication_titration | 0.317->0.324 | +0.007 | 6.397->6.488 | +1.4% | 21.7%->21.7% | +0.0 |
| Glass-2 | 5fold | realtime_alert | 0.257->0.263 | +0.006 | 4.772->4.918 | +3.1% | 8.7%->8.7% | +0.0 |
| Glass-4 | 5fold | medication_titration | 0.240->0.246 | +0.005 | 5.283->5.157 | -2.4% | 21.7%->21.7% | +0.0 |
| Glass-4 | 5fold | realtime_alert | 0.197->0.201 | +0.004 | 3.200->3.063 | -4.3% | 4.3%->4.3% | +0.0 |
| Full-18 | loo | medication_titration | 0.366->0.374 | +0.008 | 6.782->6.910 | +1.9% | 34.8%->34.8% | +0.0 |
| Full-18 | loo | realtime_alert | 0.306->0.313 | +0.007 | 3.988->4.083 | +2.4% | 17.4%->17.4% | +0.0 |
| Glass-2 | loo | medication_titration | 0.273->0.279 | +0.006 | 6.473->6.592 | +1.8% | 21.7%->21.7% | +0.0 |
| Glass-2 | loo | realtime_alert | 0.202->0.207 | +0.005 | 4.138->4.212 | +1.8% | 8.7%->8.7% | +0.0 |

**SZCORE summary:** micro-sensitivity shift range +0.003 to +0.008 (mean +0.005); micro-FA/day shift range -4.3% to +3.1% (mean +1.7%); pass-rate shift range +0.0 to +0.0 pp.


## ALI

| config | scheme | criterion | micro-sens A->B | Δ | micro-FA/day A->B | Δ% | pass-rate A->B | Δ pp |
|---|---|---|---|---|---|---|---|---|
| Best-2 | 5fold | medication_titration | 0.049->0.050 | +0.001 | 4.039->4.136 | +2.4% | 0.0%->0.0% | +0.0 |
| Best-2 | 5fold | realtime_alert | 0.027->0.028 | +0.001 | 1.930->1.989 | +3.1% | 0.0%->0.0% | +0.0 |
| Best-4 | 5fold | medication_titration | 0.082->0.084 | +0.002 | 4.470->4.581 | +2.5% | 8.7%->8.7% | +0.0 |
| Best-4 | 5fold | realtime_alert | 0.033->0.034 | +0.001 | 2.083->2.146 | +3.1% | 0.0%->0.0% | +0.0 |
| Best-7 | 5fold | medication_titration | 0.071->0.073 | +0.002 | 4.496->4.607 | +2.5% | 0.0%->0.0% | +0.0 |
| Best-7 | 5fold | realtime_alert | 0.038->0.039 | +0.001 | 1.956->2.016 | +3.1% | 0.0%->0.0% | +0.0 |
| Full-18 | 5fold | medication_titration | 0.071->0.073 | +0.002 | 4.318->4.450 | +3.1% | 4.3%->4.3% | +0.0 |
| Full-18 | 5fold | realtime_alert | 0.038->0.039 | +0.001 | 2.007->2.068 | +3.1% | 0.0%->0.0% | +0.0 |
| Glass-2 | 5fold | medication_titration | 0.044->0.045 | +0.001 | 4.417->4.526 | +2.5% | 0.0%->0.0% | +0.0 |
| Glass-2 | 5fold | realtime_alert | 0.022->0.022 | +0.000 | 1.218->1.256 | +3.1% | 0.0%->0.0% | +0.0 |
| Glass-4 | 5fold | medication_titration | 0.044->0.045 | +0.001 | 4.293->4.371 | +1.8% | 0.0%->0.0% | +0.0 |
| Glass-4 | 5fold | realtime_alert | 0.016->0.017 | +0.000 | 1.600->1.649 | +3.1% | 0.0%->0.0% | +0.0 |
| Full-18 | loo | medication_titration | 0.137->0.140 | +0.003 | 4.089->4.136 | +1.1% | 13.0%->13.0% | +0.0 |
| Full-18 | loo | realtime_alert | 0.060->0.061 | +0.001 | 2.007->2.042 | +1.7% | 0.0%->0.0% | +0.0 |
| Glass-2 | loo | medication_titration | 0.093->0.095 | +0.002 | 5.052->4.970 | -1.6% | 4.3%->4.3% | +0.0 |
| Glass-2 | loo | realtime_alert | 0.022->0.022 | +0.000 | 2.183->2.119 | -2.9% | 0.0%->0.0% | +0.0 |

**ALI summary:** micro-sensitivity shift range +0.000 to +0.003 (mean +0.001); micro-FA/day shift range -2.9% to +3.1% (mean +2.0%); pass-rate shift range +0.0 to +0.0 pp.


## Per-subject detail (the 6 subjects with non-Glass-7 files)

SzCORE, per (subject, config, scheme, criterion): hours / ref events / sens / FA per base.

| subject | config | scheme | criterion | hours A->B | ref A->B | sens A->B | FA A->B |
|---|---|---|---|---|---|---|---|
| chb13 | Best-2 | 5fold | medication_titration | 33.0->11.0 | 12->10 | 0.083->0.100 | 0.727->0.000 |
| chb13 | Best-2 | 5fold | realtime_alert | 33.0->11.0 | 12->10 | 0.000->0.000 | 0.000->0.000 |
| chb13 | Best-4 | 5fold | medication_titration | 33.0->11.0 | 12->10 | 0.000->0.000 | 0.000->0.000 |
| chb13 | Best-4 | 5fold | realtime_alert | 33.0->11.0 | 12->10 | 0.000->0.000 | 0.000->0.000 |
| chb13 | Best-7 | 5fold | medication_titration | 33.0->11.0 | 12->10 | 0.000->0.000 | 0.000->0.000 |
| chb13 | Best-7 | 5fold | realtime_alert | 33.0->11.0 | 12->10 | 0.000->0.000 | 0.000->0.000 |
| chb13 | Full-18 | 5fold | medication_titration | 33.0->11.0 | 12->10 | 0.000->0.000 | 1.455->2.182 |
| chb13 | Full-18 | 5fold | realtime_alert | 33.0->11.0 | 12->10 | 0.000->0.000 | 0.000->0.000 |
| chb13 | Full-18 | loo | medication_titration | 33.0->11.0 | 12->10 | 0.083->0.100 | 2.909->4.364 |
| chb13 | Full-18 | loo | realtime_alert | 33.0->11.0 | 12->10 | 0.000->0.000 | 1.455->2.182 |
| chb13 | Glass-2 | 5fold | medication_titration | 33.0->11.0 | 12->10 | 0.167->0.200 | 1.455->2.182 |
| chb13 | Glass-2 | 5fold | realtime_alert | 33.0->11.0 | 12->10 | 0.083->0.100 | 0.727->2.182 |
| chb13 | Glass-2 | loo | medication_titration | 33.0->11.0 | 12->10 | 0.083->0.100 | 1.455->0.000 |
| chb13 | Glass-2 | loo | realtime_alert | 33.0->11.0 | 12->10 | 0.083->0.100 | 1.455->0.000 |
| chb13 | Glass-4 | 5fold | medication_titration | 33.0->11.0 | 12->10 | 0.083->0.100 | 5.818->10.909 |
| chb13 | Glass-4 | 5fold | realtime_alert | 33.0->11.0 | 12->10 | 0.083->0.100 | 3.636->4.364 |
| chb15 | Best-2 | 5fold | medication_titration | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Best-2 | 5fold | realtime_alert | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Best-4 | 5fold | medication_titration | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Best-4 | 5fold | realtime_alert | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Best-7 | 5fold | medication_titration | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Best-7 | 5fold | realtime_alert | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Full-18 | 5fold | medication_titration | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Full-18 | 5fold | realtime_alert | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Full-18 | loo | medication_titration | 40.0->39.0 | 20->20 | 0.050->0.050 | 2.999->2.461 |
| chb15 | Full-18 | loo | realtime_alert | 40.0->39.0 | 20->20 | 0.050->0.050 | 2.399->2.461 |
| chb15 | Glass-2 | 5fold | medication_titration | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Glass-2 | 5fold | realtime_alert | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Glass-2 | loo | medication_titration | 40.0->39.0 | 20->20 | 0.000->0.000 | 6.599->6.768 |
| chb15 | Glass-2 | loo | realtime_alert | 40.0->39.0 | 20->20 | 0.000->0.000 | 2.999->3.076 |
| chb15 | Glass-4 | 5fold | medication_titration | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb15 | Glass-4 | 5fold | realtime_alert | 40.0->39.0 | 20->20 | 0.000->0.000 | 0.000->0.000 |
| chb16 | Best-2 | 5fold | medication_titration | 19.0->17.0 | 10->8 | 0.000->0.000 | 3.789->4.235 |
| chb16 | Best-2 | 5fold | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 2.526->2.824 |
| chb16 | Best-4 | 5fold | medication_titration | 19.0->17.0 | 10->8 | 0.000->0.000 | 5.053->5.647 |
| chb16 | Best-4 | 5fold | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 2.526->2.824 |
| chb16 | Best-7 | 5fold | medication_titration | 19.0->17.0 | 10->8 | 0.000->0.000 | 0.000->0.000 |
| chb16 | Best-7 | 5fold | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 0.000->0.000 |
| chb16 | Full-18 | 5fold | medication_titration | 19.0->17.0 | 10->8 | 0.000->0.000 | 0.000->0.000 |
| chb16 | Full-18 | 5fold | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 0.000->0.000 |
| chb16 | Full-18 | loo | medication_titration | 19.0->17.0 | 10->8 | 0.000->0.000 | 24.000->26.824 |
| chb16 | Full-18 | loo | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 15.158->16.941 |
| chb16 | Glass-2 | 5fold | medication_titration | 19.0->17.0 | 10->8 | 0.000->0.000 | 0.000->0.000 |
| chb16 | Glass-2 | 5fold | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 0.000->0.000 |
| chb16 | Glass-2 | loo | medication_titration | 19.0->17.0 | 10->8 | 0.100->0.125 | 11.368->12.706 |
| chb16 | Glass-2 | loo | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 5.053->5.647 |
| chb16 | Glass-4 | 5fold | medication_titration | 19.0->17.0 | 10->8 | 0.000->0.000 | 1.263->1.412 |
| chb16 | Glass-4 | 5fold | realtime_alert | 19.0->17.0 | 10->8 | 0.000->0.000 | 0.000->0.000 |
| chb17 | Best-2 | 5fold | medication_titration | 20.4->19.4 | 3->3 | 1.000->1.000 | 14.098->14.823 |
| chb17 | Best-2 | 5fold | realtime_alert | 20.4->19.4 | 3->3 | 1.000->1.000 | 9.398->9.882 |
| chb17 | Best-4 | 5fold | medication_titration | 20.4->19.4 | 3->3 | 1.000->1.000 | 21.147->19.764 |
| chb17 | Best-4 | 5fold | realtime_alert | 20.4->19.4 | 3->3 | 0.667->0.667 | 7.049->7.412 |
| chb17 | Best-7 | 5fold | medication_titration | 20.4->19.4 | 3->3 | 0.333->0.333 | 23.496->24.705 |
| chb17 | Best-7 | 5fold | realtime_alert | 20.4->19.4 | 3->3 | 0.333->0.333 | 11.748->12.353 |
| chb17 | Full-18 | 5fold | medication_titration | 20.4->19.4 | 3->3 | 0.333->0.333 | 15.272->14.823 |
| chb17 | Full-18 | 5fold | realtime_alert | 20.4->19.4 | 3->3 | 0.333->0.333 | 10.573->11.117 |
| chb17 | Full-18 | loo | medication_titration | 20.4->19.4 | 3->3 | 0.333->0.333 | 18.797->19.764 |
| chb17 | Full-18 | loo | realtime_alert | 20.4->19.4 | 3->3 | 0.333->0.333 | 12.923->13.588 |
| chb17 | Glass-2 | 5fold | medication_titration | 21.0->20.0 | 3->3 | 0.667->0.667 | 33.132->31.190 |
| chb17 | Glass-2 | 5fold | realtime_alert | 21.0->20.0 | 3->3 | 0.667->0.667 | 21.707->22.792 |
| chb17 | Glass-2 | loo | medication_titration | 21.0->20.0 | 3->3 | 0.333->0.333 | 19.422->19.194 |
| chb17 | Glass-2 | loo | realtime_alert | 21.0->20.0 | 3->3 | 0.000->0.000 | 5.712->5.998 |
| chb17 | Glass-4 | 5fold | medication_titration | 20.4->19.4 | 3->3 | 1.000->1.000 | 55.216->49.411 |
| chb17 | Glass-4 | 5fold | realtime_alert | 20.4->19.4 | 3->3 | 1.000->1.000 | 39.943->34.588 |
| chb18 | Best-2 | 5fold | medication_titration | 34.6->33.6 | 6->6 | 0.500->0.500 | 2.772->2.854 |
| chb18 | Best-2 | 5fold | realtime_alert | 34.6->33.6 | 6->6 | 0.333->0.333 | 2.079->2.141 |
| chb18 | Best-4 | 5fold | medication_titration | 34.6->33.6 | 6->6 | 0.500->0.500 | 2.772->2.854 |
| chb18 | Best-4 | 5fold | realtime_alert | 34.6->33.6 | 6->6 | 0.500->0.500 | 2.079->2.141 |
| chb18 | Best-7 | 5fold | medication_titration | 34.6->33.6 | 6->6 | 0.500->0.500 | 6.237->6.423 |
| chb18 | Best-7 | 5fold | realtime_alert | 34.6->33.6 | 6->6 | 0.167->0.167 | 1.386->1.427 |
| chb18 | Full-18 | 5fold | medication_titration | 34.6->33.6 | 6->6 | 0.833->0.833 | 6.930->7.136 |
| chb18 | Full-18 | 5fold | realtime_alert | 34.6->33.6 | 6->6 | 0.667->0.667 | 6.237->6.423 |
| chb18 | Full-18 | loo | medication_titration | 34.6->33.6 | 6->6 | 0.000->0.000 | 6.237->6.423 |
| chb18 | Full-18 | loo | realtime_alert | 34.6->33.6 | 6->6 | 0.000->0.000 | 2.772->2.854 |
| chb18 | Glass-2 | 5fold | medication_titration | 34.6->33.6 | 6->6 | 0.667->0.667 | 9.009->9.277 |
| chb18 | Glass-2 | 5fold | realtime_alert | 34.6->33.6 | 6->6 | 0.500->0.500 | 6.237->6.423 |
| chb18 | Glass-2 | loo | medication_titration | 34.6->33.6 | 6->6 | 0.000->0.000 | 5.544->5.709 |
| chb18 | Glass-2 | loo | realtime_alert | 34.6->33.6 | 6->6 | 0.000->0.000 | 2.079->2.141 |
| chb18 | Glass-4 | 5fold | medication_titration | 34.6->33.6 | 6->6 | 0.500->0.500 | 5.544->4.995 |
| chb18 | Glass-4 | 5fold | realtime_alert | 34.6->33.6 | 6->6 | 0.333->0.333 | 3.465->3.568 |
| chb19 | Best-2 | 5fold | medication_titration | 28.9->27.9 | 3->3 | 0.667->0.667 | 0.000->0.000 |
| chb19 | Best-2 | 5fold | realtime_alert | 28.9->27.9 | 3->3 | 0.667->0.667 | 0.000->0.000 |
| chb19 | Best-4 | 5fold | medication_titration | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Best-4 | 5fold | realtime_alert | 28.9->27.9 | 3->3 | 0.667->0.667 | 0.000->0.000 |
| chb19 | Best-7 | 5fold | medication_titration | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Best-7 | 5fold | realtime_alert | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Full-18 | 5fold | medication_titration | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Full-18 | 5fold | realtime_alert | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Full-18 | loo | medication_titration | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Full-18 | loo | realtime_alert | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Glass-2 | 5fold | medication_titration | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Glass-2 | 5fold | realtime_alert | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Glass-2 | loo | medication_titration | 28.9->27.9 | 3->3 | 1.000->1.000 | 1.659->1.719 |
| chb19 | Glass-2 | loo | realtime_alert | 28.9->27.9 | 3->3 | 1.000->1.000 | 0.000->0.000 |
| chb19 | Glass-4 | 5fold | medication_titration | 28.9->27.9 | 3->3 | 0.667->0.667 | 0.000->0.000 |
| chb19 | Glass-4 | 5fold | realtime_alert | 28.9->27.9 | 3->3 | 0.667->0.667 | 0.000->0.000 |
