import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_data = """
Running inner cross-validation for outer fold 1/5 with 48 models M_i...
Running M_1 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.01}...   finished in 3.31 seconds | mean inner error = 0.5731
Running M_2 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.5}...    finished in 3.42 seconds | mean inner error = 0.9398
Running M_3 with params {'lr': 0.0001, 'epochs': 100, 'reg': 1}...      finished in 3.31 seconds | mean inner error = 0.9517
Running M_4 with params {'lr': 0.0001, 'epochs': 100, 'reg': 10}...     finished in 3.02 seconds | mean inner error = 0.9970
Running M_5 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.01}...   finished in 5.78 seconds | mean inner error = 0.5714
Running M_6 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.5}...    finished in 4.64 seconds | mean inner error = 0.9326
Running M_7 with params {'lr': 0.0001, 'epochs': 200, 'reg': 1}...      finished in 11.02 seconds | mean inner error = 0.9591
Running M_8 with params {'lr': 0.0001, 'epochs': 200, 'reg': 10}...     finished in 11.35 seconds | mean inner error = 0.9875
Running M_9 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.01}...   finished in 28.94 seconds | mean inner error = 0.5673
Running M_10 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.5}...   finished in 16.96 seconds | mean inner error = 0.9275
Running M_11 with params {'lr': 0.0001, 'epochs': 500, 'reg': 1}...     finished in 9.30 seconds | mean inner error = 0.9542
Running M_12 with params {'lr': 0.0001, 'epochs': 500, 'reg': 10}...    finished in 9.59 seconds | mean inner error = 0.9709
Running M_13 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.01}...   finished in 1.84 seconds | mean inner error = 0.5618
Running M_14 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.5}...    finished in 1.92 seconds | mean inner error = 0.9398
Running M_15 with params {'lr': 0.001, 'epochs': 100, 'reg': 1}...      finished in 1.82 seconds | mean inner error = 0.9514
Running M_16 with params {'lr': 0.001, 'epochs': 100, 'reg': 10}...     finished in 1.74 seconds | mean inner error = 0.9973
Running M_17 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.01}...   finished in 3.43 seconds | mean inner error = 0.5474
Running M_18 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.5}...    finished in 3.48 seconds | mean inner error = 0.9350
Running M_19 with params {'lr': 0.001, 'epochs': 200, 'reg': 1}...      finished in 3.51 seconds | mean inner error = 0.9602
Running M_20 with params {'lr': 0.001, 'epochs': 200, 'reg': 10}...     finished in 3.57 seconds | mean inner error = 0.9873
Running M_21 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.01}...   finished in 9.48 seconds | mean inner error = 0.5118
Running M_22 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.5}...    finished in 9.52 seconds | mean inner error = 0.9335
Running M_23 with params {'lr': 0.001, 'epochs': 500, 'reg': 1}...      finished in 10.19 seconds | mean inner error = 0.9588
Running M_24 with params {'lr': 0.001, 'epochs': 500, 'reg': 10}...     finished in 26.61 seconds | mean inner error = 0.9688
Running M_25 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.01}...    finished in 5.95 seconds | mean inner error = 0.4546
Running M_26 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.5}...     finished in 5.87 seconds | mean inner error = 0.9311
Running M_27 with params {'lr': 0.01, 'epochs': 100, 'reg': 1}...       finished in 5.86 seconds | mean inner error = 0.9499
Running M_28 with params {'lr': 0.01, 'epochs': 100, 'reg': 10}...      finished in 5.99 seconds | mean inner error = 0.9978
Running M_29 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.01}...    finished in 11.99 seconds | mean inner error = 0.3794
Running M_30 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.5}...     finished in 11.33 seconds | mean inner error = 0.9321
Running M_31 with params {'lr': 0.01, 'epochs': 200, 'reg': 1}...       finished in 10.97 seconds | mean inner error = 0.9466
Running M_32 with params {'lr': 0.01, 'epochs': 200, 'reg': 10}...      finished in 10.75 seconds | mean inner error = 0.9813
Running M_33 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.01}...    finished in 27.87 seconds | mean inner error = 0.3109
Running M_34 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.5}...     finished in 28.97 seconds | mean inner error = 0.9319
Running M_35 with params {'lr': 0.01, 'epochs': 500, 'reg': 1}...       finished in 28.75 seconds | mean inner error = 0.9430
Running M_36 with params {'lr': 0.01, 'epochs': 500, 'reg': 10}...      finished in 30.18 seconds | mean inner error = 0.9633
Running M_37 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.01}...    finished in 5.58 seconds | mean inner error = 0.3110
Running M_38 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.5}...     finished in 5.35 seconds | mean inner error = 0.9388
Running M_39 with params {'lr': 0.05, 'epochs': 100, 'reg': 1}...       finished in 4.91 seconds | mean inner error = 0.9417
Running M_40 with params {'lr': 0.05, 'epochs': 100, 'reg': 10}...      finished in 5.92 seconds | mean inner error = 0.9982
Running M_41 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.01}...    finished in 11.56 seconds | mean inner error = 0.2945
Running M_42 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.5}...     finished in 11.82 seconds | mean inner error = 0.9220
Running M_43 with params {'lr': 0.05, 'epochs': 200, 'reg': 1}...       finished in 11.87 seconds | mean inner error = 0.9451
Running M_44 with params {'lr': 0.05, 'epochs': 200, 'reg': 10}...      finished in 11.64 seconds | mean inner error = 0.9865
Running M_45 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.01}...    finished in 27.28 seconds | mean inner error = 0.2918
Running M_46 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.5}...     finished in 28.95 seconds | mean inner error = 0.9273
Running M_47 with params {'lr': 0.05, 'epochs': 500, 'reg': 1}...       finished in 29.06 seconds | mean inner error = 0.9385
Running M_48 with params {'lr': 0.05, 'epochs': 500, 'reg': 10}...      finished in 28.80 seconds | mean inner error = 0.9318

Running inner cross-validation for outer fold 2/5 with 48 models M_i...
Running M_1 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.01}...   finished in 5.64 seconds | mean inner error = 0.5688
Running M_2 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.5}...    finished in 5.53 seconds | mean inner error = 0.9323
Running M_3 with params {'lr': 0.0001, 'epochs': 100, 'reg': 1}...      finished in 5.77 seconds | mean inner error = 0.9611
Running M_4 with params {'lr': 0.0001, 'epochs': 100, 'reg': 10}...     finished in 5.31 seconds | mean inner error = 0.9960
Running M_5 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.01}...   finished in 10.97 seconds | mean inner error = 0.5678
Running M_6 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.5}...    finished in 11.57 seconds | mean inner error = 0.9357
Running M_7 with params {'lr': 0.0001, 'epochs': 200, 'reg': 1}...      finished in 12.28 seconds | mean inner error = 0.9541
Running M_8 with params {'lr': 0.0001, 'epochs': 200, 'reg': 10}...     finished in 11.74 seconds | mean inner error = 0.9842
Running M_9 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.01}...   finished in 28.91 seconds | mean inner error = 0.5646
Running M_10 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.5}...   finished in 29.10 seconds | mean inner error = 0.9457
Running M_11 with params {'lr': 0.0001, 'epochs': 500, 'reg': 1}...     finished in 22.94r = 0.9567
Running M_12 with params {'lr': 0.0001, 'epochs': 500, 'reg': 10}...    finished in 8.78 seconds | mean inner error = 0.9677
Running M_13 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.01}...   finished in 1.77 seconds | mean inner error = 0.5571
Running M_14 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.5}...    finished in 1.79 seconds | mean inner error = 0.9310
Running M_15 with params {'lr': 0.001, 'epochs': 100, 'reg': 1}...      finished in 1.90 seconds | mean inner error = 0.9599
Running M_16 with params {'lr': 0.001, 'epochs': 100, 'reg': 10}...     finished in 1.79 seconds | mean inner error = 0.9961
Running M_17 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.01}...   finished in 3.73 seconds | mean inner error = 0.5446
Running M_18 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.5}...    finished in 4.34 seconds | mean inner error = 0.9368
Running M_19 with params {'lr': 0.001, 'epochs': 200, 'reg': 1}...      finished in 4.13 seconds | mean inner error = 0.9527
Running M_20 with params {'lr': 0.001, 'epochs': 200, 'reg': 10}...     finished in 3.74 seconds | mean inner error = 0.9825
Running M_21 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.01}...   finished in 12.42 seconds | mean inner error = 0.5102
Running M_22 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.5}...    finished in 29.16 seconds | mean inner error = 0.9449
Running M_23 with params {'lr': 0.001, 'epochs': 500, 'reg': 1}...      finished in 16.21 seconds | mean inner error = 0.9599
Running M_24 with params {'lr': 0.001, 'epochs': 500, 'reg': 10}...     finished in 14.88 seconds | mean inner error = 0.9628
Running M_25 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.01}...    finished in 2.76 seconds | mean inner error = 0.4539
Running M_26 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.5}...     finished in 3.02 seconds | mean inner error = 0.9417
Running M_27 with params {'lr': 0.01, 'epochs': 100, 'reg': 1}...       finished in 3.08 seconds | mean inner error = 0.9603
Running M_28 with params {'lr': 0.01, 'epochs': 100, 'reg': 10}...      finished in 3.88 seconds | mean inner error = 0.9968
Running M_29 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.01}...    finished in 10.60 seconds | mean inner error = 0.3793
Running M_30 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.5}...     finished in 4.75 seconds | mean inner error = 0.9263
Running M_31 with params {'lr': 0.01, 'epochs': 200, 'reg': 1}...       finished in 4.31 seconds | mean inner error = 0.9523
Running M_32 with params {'lr': 0.01, 'epochs': 200, 'reg': 10}...      finished in 4.35 seconds | mean inner error = 0.9825
Running M_33 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.01}...    finished in 15.40 seconds | mean inner error = 0.3116
Running M_34 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.5}...     finished in 14.13 seconds | mean inner error = 0.9277
Running M_35 with params {'lr': 0.01, 'epochs': 500, 'reg': 1}...       finished in 14.12 seconds | mean inner error = 0.9529
Running M_36 with params {'lr': 0.01, 'epochs': 500, 'reg': 10}...      finished in 9.96 seconds | mean inner error = 0.9526
Running M_37 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.01}...    finished in 2.19 seconds | mean inner error = 0.3112
Running M_38 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.5}...     finished in 2.05 seconds | mean inner error = 0.9295
Running M_39 with params {'lr': 0.05, 'epochs': 100, 'reg': 1}...       finished in 2.15 seconds | mean inner error = 0.9507
Running M_40 with params {'lr': 0.05, 'epochs': 100, 'reg': 10}...      finished in 2.13 seconds | mean inner error = 0.9976
Running M_41 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.01}...    finished in 3.99 seconds | mean inner error = 0.2940
Running M_42 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.5}...     finished in 3.95 seconds | mean inner error = 0.9250
Running M_43 with params {'lr': 0.05, 'epochs': 200, 'reg': 1}...       finished in 4.13 seconds | mean inner error = 0.9427
Running M_44 with params {'lr': 0.05, 'epochs': 200, 'reg': 10}...      finished in 4.73 seconds | mean inner error = 0.9856
Running M_45 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.01}...    finished in 12.55 seconds | mean inner error = 0.2904
Running M_46 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.5}...     finished in 12.71 seconds | mean inner error = 0.9121
Running M_47 with params {'lr': 0.05, 'epochs': 500, 'reg': 1}...       finished in 15.64 seconds | mean inner error = 0.9375
Running M_48 with params {'lr': 0.05, 'epochs': 500, 'reg': 10}...      finished in 17.01 seconds | mean inner error = 0.9331

Running inner cross-validation for outer fold 3/5 with 48 models M_i...
Running M_1 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.01}...   finished in 5.88 seconds | mean inner error = 0.5843
Running M_2 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.5}...    finished in 5.92 seconds | mean inner error = 0.9356
Running M_3 with params {'lr': 0.0001, 'epochs': 100, 'reg': 1}...      finished in 5.77 seconds | mean inner error = 0.9503
Running M_4 with params {'lr': 0.0001, 'epochs': 100, 'reg': 10}...     finished in 5.59 seconds | mean inner error = 0.9955
Running M_5 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.01}...   finished in 11.59 seconds | mean inner error = 0.5840
Running M_6 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.5}...    finished in 12.00 seconds | mean inner error = 0.9361
Running M_7 with params {'lr': 0.0001, 'epochs': 200, 'reg': 1}...      finished in 11.33 seconds | mean inner error = 0.9530
Running M_8 with params {'lr': 0.0001, 'epochs': 200, 'reg': 10}...     finished in 11.75 seconds | mean inner error = 0.9870
Running M_9 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.01}...   finished in 28.01 seconds | mean inner error = 0.5800
Running M_10 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.5}...   finished in 27.96 seconds | mean inner error = 0.9338
Running M_11 with params {'lr': 0.0001, 'epochs': 500, 'reg': 1}...     finished in 24.21 seconds | mean inner error = 0.9458
Running M_12 with params {'lr': 0.0001, 'epochs': 500, 'reg': 10}...    finished in 29.18 seconds | mean inner error = 0.9663
Running M_13 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.01}...   finished in 5.89 seconds | mean inner error = 0.5709
Running M_14 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.5}...    finished in 5.53 seconds | mean inner error = 0.9355
Running M_15 with params {'lr': 0.001, 'epochs': 100, 'reg': 1}...      finished in 5.25 seconds | mean inner error = 0.9505
Running M_16 with params {'lr': 0.001, 'epochs': 100, 'reg': 10}...     finished in 5.58 seconds | mean inner error = 0.9959
Running M_17 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.01}...   finished in 11.78 seconds | mean inner error = 0.5570
Running M_18 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.5}...    finished in 10.99 seconds | mean inner error = 0.9328
Running M_19 with params {'lr': 0.001, 'epochs': 200, 'reg': 1}...      finished in 11.43 seconds | mean inner error = 0.9527
Running M_20 with params {'lr': 0.001, 'epochs': 200, 'reg': 10}...     finished in 10.41 seconds | mean inner error = 0.9865
Running M_21 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.01}...   finished in 28.26 seconds | mean inner error = 0.5177
Running M_22 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.5}...    finished in 13.60 seconds | mean inner error = 0.9330
Running M_23 with params {'lr': 0.001, 'epochs': 500, 'reg': 1}...      finished in 13.05 seconds | mean inner error = 0.9470
Running M_24 with params {'lr': 0.001, 'epochs': 500, 'reg': 10}...     finished in 12.18 seconds | mean inner error = 0.9662
Running M_25 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.01}...    finished in 2.23 seconds | mean inner error = 0.4580
Running M_26 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.5}...     finished in 2.19 seconds | mean inner error = 0.9308
Running M_27 with params {'lr': 0.01, 'epochs': 100, 'reg': 1}...       finished in 2.34 seconds | mean inner error = 0.9550
Running M_28 with params {'lr': 0.01, 'epochs': 100, 'reg': 10}...      finished in 2.60 seconds | mean inner error = 0.9967
Running M_29 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.01}...    finished in 5.67 seconds | mean inner error = 0.3810
Running M_30 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.5}...     finished in 5.01 seconds | mean inner error = 0.9263
Running M_31 with params {'lr': 0.01, 'epochs': 200, 'reg': 1}...       finished in 5.23 seconds | mean inner error = 0.9501
Running M_32 with params {'lr': 0.01, 'epochs': 200, 'reg': 10}...      finished in 5.02 seconds | mean inner error = 0.9838
Running M_33 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.01}...    finished in 14.03 seconds | mean inner error = 0.3140
Running M_34 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.5}...     finished in 15.33 seconds | mean inner error = 0.9292
Running M_35 with params {'lr': 0.01, 'epochs': 500, 'reg': 1}...       finished in 24.33 seconds | mean inner error = 0.9419
Running M_36 with params {'lr': 0.01, 'epochs': 500, 'reg': 10}...      finished in 27.65 seconds | mean inner error = 0.9640
Running M_37 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.01}...    finished in 5.51 seconds | mean inner error = 0.3127
Running M_38 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.5}...     finished in 6.28 seconds | mean inner error = 0.9181
Running M_39 with params {'lr': 0.05, 'epochs': 100, 'reg': 1}...       finished in 5.97 seconds | mean inner error = 0.9445
Running M_40 with params {'lr': 0.05, 'epochs': 100, 'reg': 10}...      finished in 6.39 seconds | mean inner error = 0.9982
Running M_41 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.01}...    finished in 11.76 seconds | mean inner error = 0.2948
Running M_42 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.5}...     finished in 11.31 seconds | mean inner error = 0.9245
Running M_43 with params {'lr': 0.05, 'epochs': 200, 'reg': 1}...       finished in 11.48 seconds | mean inner error = 0.9432
Running M_44 with params {'lr': 0.05, 'epochs': 200, 'reg': 10}...      finished in 11.35 seconds | mean inner error = 0.9859
Running M_45 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.01}...    finished in 28.42 seconds | mean inner error = 0.2927
Running M_46 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.5}...     finished in 28.47 seconds | mean inner error = 0.9139
Running M_47 with params {'lr': 0.05, 'epochs': 500, 'reg': 1}...       finished in 20.44 seconds | mean inner error = 0.9490
Running M_48 with params {'lr': 0.05, 'epochs': 500, 'reg': 10}...      finished in 27.45 seconds | mean inner error = 0.9493

Running inner cross-validation for outer fold 4/5 with 48 models M_i...
Running M_1 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.01}...   finished in 5.39 seconds | mean inner error = 0.5942
Running M_2 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.5}...    finished in 5.81 seconds | mean inner error = 0.9295
Running M_3 with params {'lr': 0.0001, 'epochs': 100, 'reg': 1}...      finished in 5.35 seconds | mean inner error = 0.9531
Running M_4 with params {'lr': 0.0001, 'epochs': 100, 'reg': 10}...     finished in 5.60 seconds | mean inner error = 0.9952
Running M_5 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.01}...   finished in 11.62 seconds | mean inner error = 0.5929
Running M_6 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.5}...    finished in 12.61 seconds | mean inner error = 0.9334
Running M_7 with params {'lr': 0.0001, 'epochs': 200, 'reg': 1}...      finished in 11.23 seconds | mean inner error = 0.9498
Running M_8 with params {'lr': 0.0001, 'epochs': 200, 'reg': 10}...     finished in 12.04 seconds | mean inner error = 0.9857
Running M_9 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.01}...   finished in 29.15 seconds | mean inner error = 0.5883
Running M_10 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.5}...   finished in 28.72 seconds | mean inner error = 0.9273
Running M_11 with params {'lr': 0.0001, 'epochs': 500, 'reg': 1}...     finished in 27.08 seconds | mean inner error = 0.9509
Running M_12 with params {'lr': 0.0001, 'epochs': 500, 'reg': 10}...    finished in 29.55 seconds | mean inner error = 0.9637
Running M_13 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.01}...   finished in 6.38 seconds | mean inner error = 0.5809
Running M_14 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.5}...    finished in 5.57 seconds | mean inner error = 0.9287
Running M_15 with params {'lr': 0.001, 'epochs': 100, 'reg': 1}...      finished in 5.91 seconds | mean inner error = 0.9515
Running M_16 with params {'lr': 0.001, 'epochs': 100, 'reg': 10}...     finished in 5.67 seconds | mean inner error = 0.9954
Running M_17 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.01}...   finished in 11.16 seconds | mean inner error = 0.5617
Running M_18 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.5}...    finished in 8.00 seconds | mean inner error = 0.9343
Running M_19 with params {'lr': 0.001, 'epochs': 200, 'reg': 1}...      finished in 3.89 seconds | mean inner error = 0.9509
Running M_20 with params {'lr': 0.001, 'epochs': 200, 'reg': 10}...     finished in 4.01 seconds | mean inner error = 0.9854
Running M_21 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.01}...   finished in 13.42 seconds | mean inner error = 0.5208
Running M_22 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.5}...    finished in 18.83 seconds | mean inner error = 0.9375
Running M_23 with params {'lr': 0.001, 'epochs': 500, 'reg': 1}...      finished in 14.10 seconds | mean inner error = 0.9548
Running M_24 with params {'lr': 0.001, 'epochs': 500, 'reg': 10}...     finished in 15.62 seconds | mean inner error = 0.9665
Running M_25 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.01}...    finished in 2.53 seconds | mean inner error = 0.4566
Running M_26 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.5}...     finished in 2.57 seconds | mean inner error = 0.9290
Running M_27 with params {'lr': 0.01, 'epochs': 100, 'reg': 1}...       finished in 2.74 seconds | mean inner error = 0.9598
Running M_28 with params {'lr': 0.01, 'epochs': 100, 'reg': 10}...      finished in 2.87 seconds | mean inner error = 0.9973
Running M_29 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.01}...    finished in 11.20 seconds | mean inner error = 0.3819
Running M_30 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.5}...     finished in 12.32 seconds | mean inner error = 0.9299
Running M_31 with params {'lr': 0.01, 'epochs': 200, 'reg': 1}...       finished in 11.77 seconds | mean inner error = 0.9517
Running M_32 with params {'lr': 0.01, 'epochs': 200, 'reg': 10}...      finished in 12.04 seconds | mean inner error = 0.9872
Running M_33 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.01}...    finished in 28.89 seconds | mean inner error = 0.3133
Running M_34 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.5}...     finished in 28.87 seconds | mean inner error = 0.9318
Running M_35 with params {'lr': 0.01, 'epochs': 500, 'reg': 1}...       finished in 28.72 seconds | mean inner error = 0.9423
Running M_36 with params {'lr': 0.01, 'epochs': 500, 'reg': 10}...      finished in 28.86 seconds | mean inner error = 0.9624
Running M_37 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.01}...    finished in 5.65 seconds | mean inner error = 0.3123
Running M_38 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.5}...     finished in 5.77 seconds | mean inner error = 0.9113
Running M_39 with params {'lr': 0.05, 'epochs': 100, 'reg': 1}...       finished in 5.42 seconds | mean inner error = 0.9348
Running M_40 with params {'lr': 0.05, 'epochs': 100, 'reg': 10}...      finished in 5.74 seconds | mean inner error = 0.9981
Running M_41 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.01}...    finished in 10.89 seconds | mean inner error = 0.2943
Running M_42 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.5}...     finished in 10.65 seconds | mean inner error = 0.9185
Running M_43 with params {'lr': 0.05, 'epochs': 200, 'reg': 1}...       finished in 11.87 seconds | mean inner error = 0.9357
Running M_44 with params {'lr': 0.05, 'epochs': 200, 'reg': 10}...      finished in 11.96 seconds | mean inner error = 0.9874
Running M_45 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.01}...    finished in 27.82 seconds | mean inner error = 0.2925
Running M_46 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.5}...     finished in 22.76 seconds | mean inner error = 0.9103
Running M_47 with params {'lr': 0.05, 'epochs': 500, 'reg': 1}...       finished in 28.17 seconds | mean inner error = 0.9313
Running M_48 with params {'lr': 0.05, 'epochs': 500, 'reg': 10}...      finished in 27.80 seconds | mean inner error = 0.9372

Running inner cross-validation for outer fold 5/5 with 48 models M_i...
Running M_1 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.01}...   finished in 5.91 seconds | mean inner error = 0.5863
Running M_2 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.5}...    finished in 6.00 seconds | mean inner error = 0.9360
Running M_3 with params {'lr': 0.0001, 'epochs': 100, 'reg': 1}...      finished in 6.20 seconds | mean inner error = 0.9520
Running M_4 with params {'lr': 0.0001, 'epochs': 100, 'reg': 10}...     finished in 6.20 seconds | mean inner error = 0.9971
Running M_5 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.01}...   finished in 74.76 seconds | mean inner error = 0.5847
Running M_6 with params {'lr': 0.0001, 'epochs': 200, 'reg': 0.5}...    finished in 208.54 seconds | mean inner error = 0.9379
Running M_7 with params {'lr': 0.0001, 'epochs': 200, 'reg': 1}...      finished in 79.71 seconds | mean inner error = 0.9545
Running M_8 with params {'lr': 0.0001, 'epochs': 200, 'reg': 10}...     finished in 101.17 seconds | mean inner error = 0.9907
Running M_9 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.01}...   finished in 46.73 seconds | mean inner error = 0.5797
Running M_10 with params {'lr': 0.0001, 'epochs': 500, 'reg': 0.5}...   finished in 12.42 seconds | mean inner error = 0.9239
Running M_11 with params {'lr': 0.0001, 'epochs': 500, 'reg': 1}...     finished in 329.74 seconds | mean inner error = 0.9481
Running M_12 with params {'lr': 0.0001, 'epochs': 500, 'reg': 10}...    finished in 14.98 seconds | mean inner error = 0.9667
Running M_13 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.01}...   finished in 2.45 seconds | mean inner error = 0.5729
Running M_14 with params {'lr': 0.001, 'epochs': 100, 'reg': 0.5}...    finished in 2.71 seconds | mean inner error = 0.9368
Running M_15 with params {'lr': 0.001, 'epochs': 100, 'reg': 1}...      finished in 2.37 seconds | mean inner error = 0.9524
Running M_16 with params {'lr': 0.001, 'epochs': 100, 'reg': 10}...     finished in 2.37 seconds | mean inner error = 0.9970
Running M_17 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.01}...   finished in 4.50 seconds | mean inner error = 0.5603
Running M_18 with params {'lr': 0.001, 'epochs': 200, 'reg': 0.5}...    finished in 4.25 seconds | mean inner error = 0.9360
Running M_19 with params {'lr': 0.001, 'epochs': 200, 'reg': 1}...      finished in 3.71 seconds | mean inner error = 0.9561
Running M_20 with params {'lr': 0.001, 'epochs': 200, 'reg': 10}...     finished in 3.55 seconds | mean inner error = 0.9897
Running M_21 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.01}...   finished in 8.69 seconds | mean inner error = 0.5223
Running M_22 with params {'lr': 0.001, 'epochs': 500, 'reg': 0.5}...    finished in 8.95 seconds | mean inner error = 0.9253
Running M_23 with params {'lr': 0.001, 'epochs': 500, 'reg': 1}...      finished in 9.97 seconds | mean inner error = 0.9459
Running M_24 with params {'lr': 0.001, 'epochs': 500, 'reg': 10}...     finished in 9.20 seconds | mean inner error = 0.9625
Running M_25 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.01}...    finished in 1.85 seconds | mean inner error = 0.4620
Running M_26 with params {'lr': 0.01, 'epochs': 100, 'reg': 0.5}...     finished in 1.94 seconds | mean inner error = 0.9433
Running M_27 with params {'lr': 0.01, 'epochs': 100, 'reg': 1}...       finished in 1.85 seconds | mean inner error = 0.9504
Running M_28 with params {'lr': 0.01, 'epochs': 100, 'reg': 10}...      finished in 1.83 seconds | mean inner error = 0.9974
Running M_29 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.01}...    finished in 3.64 seconds | mean inner error = 0.3824
Running M_30 with params {'lr': 0.01, 'epochs': 200, 'reg': 0.5}...     finished in 3.71 seconds | mean inner error = 0.9366
Running M_31 with params {'lr': 0.01, 'epochs': 200, 'reg': 1}...       finished in 3.59 seconds | mean inner error = 0.9577
Running M_32 with params {'lr': 0.01, 'epochs': 200, 'reg': 10}...      finished in 10.80 seconds | mean inner error = 0.9856
Running M_33 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.01}...    finished in 28.76 seconds | mean inner error = 0.3122
Running M_34 with params {'lr': 0.01, 'epochs': 500, 'reg': 0.5}...     finished in 26.67 seconds | mean inner error = 0.9303
Running M_35 with params {'lr': 0.01, 'epochs': 500, 'reg': 1}...       finished in 27.27 seconds | mean inner error = 0.9521
Running M_36 with params {'lr': 0.01, 'epochs': 500, 'reg': 10}...      finished in 27.91 seconds | mean inner error = 0.9516
Running M_37 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.01}...    finished in 5.33 seconds | mean inner error = 0.3117
Running M_38 with params {'lr': 0.05, 'epochs': 100, 'reg': 0.5}...     finished in 5.41 seconds | mean inner error = 0.9191
Running M_39 with params {'lr': 0.05, 'epochs': 100, 'reg': 1}...       finished in 5.64 seconds | mean inner error = 0.9503
Running M_40 with params {'lr': 0.05, 'epochs': 100, 'reg': 10}...      finished in 5.43 seconds | mean inner error = 0.9978
Running M_41 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.01}...    finished in 10.75 seconds | mean inner error = 0.2931
Running M_42 with params {'lr': 0.05, 'epochs': 200, 'reg': 0.5}...     finished in 10.49 seconds | mean inner error = 0.9288
Running M_43 with params {'lr': 0.05, 'epochs': 200, 'reg': 1}...       finished in 11.56 seconds | mean inner error = 0.9427
Running M_44 with params {'lr': 0.05, 'epochs': 200, 'reg': 10}...      finished in 10.97 seconds | mean inner error = 0.9891
Running M_45 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.01}...    finished in 26.67 seconds | mean inner error = 0.2921
Running M_46 with params {'lr': 0.05, 'epochs': 500, 'reg': 0.5}...     finished in 26.35 seconds | mean inner error = 0.9098
Running M_47 with params {'lr': 0.05, 'epochs': 500, 'reg': 1}...       finished in 26.73 seconds | mean inner error = 0.9463
Running M_48 with params {'lr': 0.05, 'epochs': 500, 'reg': 10}...      finished in 25.56 seconds | mean inner error = 0.9420
"""

results = []
current_fold = 0
# Regex to match parameters and errors
# Example line: Running M_1 with params {'lr': 0.0001, 'epochs': 100, 'reg': 0.01}...   finished in 3.31 seconds | mean inner error = 0.5731
pattern = re.compile(r"Running M_(\d+) with params (\{.*\}).*mean inner error = ([\d\.]+)")

for line in log_data.split('\n'):
    if "outer fold" in line:
        current_fold = int(re.search(r"fold (\d)/5", line).group(1))
    
    match = pattern.search(line)
    if match:
        m_id = int(match.group(1))
        params = eval(match.group(2))
        error = float(match.group(3))
        results.append({
            'fold': current_fold,
            'm_id': m_id,
            'lr': params['lr'],
            'epochs': params['epochs'],
            'reg': params['reg'],
            'error': error
        })

df = pd.DataFrame(results)

# Average error for each model (m_id) across all folds
avg_df = df.groupby(['lr', 'epochs', 'reg']).agg({'error': 'mean'}).reset_index()

# Plot 1: Error vs Reg (for different Learning Rates, fix epochs=500)
plt.figure(figsize=(10, 6))
subset1 = avg_df[avg_df['epochs'] == 500]
sns.lineplot(data=subset1, x='reg', y='error', hue='lr', marker='o')
plt.title('Mean Inner Error vs Regularization (Epochs=500)')
plt.xscale('log')
plt.grid(True)
plt.savefig('error_vs_reg.png')
plt.close()

# Plot 2: Error vs Epochs (for different Learning Rates, fix reg=0.01)
plt.figure(figsize=(10, 6))
subset2 = avg_df[avg_df['reg'] == 0.01]
sns.lineplot(data=subset2, x='epochs', y='error', hue='lr', marker='o')
plt.title('Mean Inner Error vs Epochs (Reg=0.01)')
plt.grid(True)
plt.savefig('error_vs_epochs.png')
plt.close()

# Plot 3: Error vs Learning Rate (for different Regularization, fix epochs=500)
plt.figure(figsize=(10, 6))
subset3 = avg_df[avg_df['epochs'] == 500]
sns.lineplot(data=subset3, x='lr', y='error', hue='reg', marker='o')
plt.title('Mean Inner Error vs Learning Rate (Epochs=500)')
plt.xscale('log')
plt.grid(True)
plt.savefig('error_vs_lr.png')
plt.close()

# Plot 4: Heatmap (Fixing Epochs=500, showing lr vs reg)
plt.figure(figsize=(10, 6))
hm4 = avg_df[avg_df['epochs'] == 500].pivot(index='lr', columns='reg', values='error')
if hm4.size:
    sns.heatmap(hm4, annot=True, fmt=".4f", cmap='RdYlGn_r')
    plt.title('Heatmap: Mean Inner Error (Epochs=500)')
    plt.savefig('error_heatmap.png')
else:
    print("Skipping error_heatmap.png: no rows with epochs==500")
plt.close()

# Plot 5: Heatmap (Fixing reg=0.01, showing lr vs epochs) — log uses reg in {0.01, 0.5, 1, 10}, not 0.0001
plt.figure(figsize=(10, 6))
hm5 = avg_df[avg_df['reg'] == 0.01].pivot(index='lr', columns='epochs', values='error')
if hm5.size:
    sns.heatmap(hm5, annot=True, fmt=".4f", cmap='RdYlGn_r')
    plt.title('Heatmap: Mean Inner Error (Reg=0.01)')
    plt.savefig('error_heatmap_reg_0.01.png')
else:
    print("Skipping error_heatmap_reg_0.01.png: no rows with reg==0.01")
plt.close()

print(avg_df.head())
avg_df.to_csv('average_inner_errors.csv', index=False)