# Chaos suite: 12/12 passed

| test                           | result   | detail                                                                                                                 |   seconds |
|:-------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------|----------:|
| battery_outage_mid_discharge   | PASS     | re-planned on the next tick without BESS-1; unserved=0.00 MW; violations=0                                             |       0.9 |
| silent_battery_fault_recovery  | PASS     | first command at tick 2, fault detected at tick 2; inspection request #1 (approved); battery back in service: True     |       1   |
| line_congestion                | PASS     | max L-NORTH flow 35.0 MW <= 35 MW; 34.8 MWh stored; violations=0                                                       |       0.7 |
| price_spike                    | PASS     | discharged 49.2 MWh during spike vs 0.0 MWh without; situations=['normal', 'price_spike']                              |       1   |
| two_simultaneous_failures      | PASS     | no crash; violations=0; situation=asset_failure; human alert raised: True; unserved=0.00 MWh                           |       0.8 |
| sensor_dropout                 | PASS     | 4 ticks served from last-known-good values ['demand', 'price']; no crash                                               |       0.3 |
| stale_forecast                 | PASS     | situation=high_uncertainty; reserves raised to {'BESS-1': 30.0, 'BESS-2': 30.0}                                        |       0.1 |
| malformed_llm_output           | PASS     | rejected twice (weights must sum to 1 (got 1.300); reliability weight may never drop b...), fell back to rules profile |       0.1 |
| injection_in_pdf               | PASS     | 5 injection patterns flagged; constraints identical (1); strategy and dispatch unchanged: True                         |       0.5 |
| storm_with_advance_warning     | PASS     | SOC at storm onset: GridMind 124 MWh vs MPC 124 MWh; unserved 0.00 vs 23.29 MWh; cost Rs 104.0 vs 149.7 lakh           |       3.2 |
| demand_surge                   | PASS     | +30 MW absorbed; unserved 0.00 MWh; violations=0                                                                       |       0.5 |
| maintenance_planning_and_delay | PASS     | proposed WND-2 window 23 Oct 09:00 (lost value Rs 64,774); storm overlap -> delay requested and approved: True         |       0.1 |
