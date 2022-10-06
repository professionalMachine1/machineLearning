import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble

'''


Задача: Разработать модель, которая способна предсказывать летальность госпитализированных пациентов в зависимости 
от различных причин госпитализации



'''

'''

Считываем данные

'''

fileway = 'C:\\Users\\HOME\\Desktop\\dataset.csv'
data = pd.read_csv(fileway, sep = ',')

'''

Вычисляем основные характеристики

'''

#Среднее, стандартное отклонение и другие

data.describe(include = None)
Out  [1]: 
        encounter_id  ...  hospital_death
count   91713.000000  ...    91713.000000
mean    65606.079280  ...        0.086302
std     37795.088538  ...        0.280811
min         1.000000  ...        0.000000
25%     32852.000000  ...        0.000000
50%     65665.000000  ...        0.000000
75%     98342.000000  ...        0.000000
max    131051.000000  ...        1.000000

[8 rows x 77 columns]

#Корреляция

pd.set_option('display.max_columns', None)
data.corr(numeric_only=True)
Out  [2]: 
                             encounter_id  patient_id  hospital_id       age  \
encounter_id                     1.000000   -0.009575    -0.004532 -0.003990   
patient_id                      -0.009575    1.000000    -0.007075  0.006343   
hospital_id                     -0.004532   -0.007075     1.000000 -0.008673   
age                             -0.003990    0.006343    -0.008673  1.000000   
bmi                             -0.001068   -0.001380     0.012874 -0.087077   
                                  ...         ...          ...       ...   
immunosuppression               -0.001707    0.000548     0.000146  0.025007   
leukemia                        -0.002172    0.000844    -0.002992  0.030310   
lymphoma                        -0.001176    0.002333     0.002588  0.023335   
solid_tumor_with_metastasis     -0.004646   -0.003065    -0.004730  0.025924   
hospital_death                  -0.005243    0.004877    -0.001255  0.111017   

                                  bmi  elective_surgery    height    icu_id  \
encounter_id                -0.001068          0.002036 -0.005400 -0.000992   
patient_id                  -0.001380          0.001387  0.002902 -0.001770   
hospital_id                  0.012874          0.052123  0.027895  0.004526   
age                         -0.087077          0.067320 -0.109937 -0.024257   
bmi                          1.000000          0.015921 -0.056316  0.001403   
                              ...               ...       ...       ...   
immunosuppression           -0.031144         -0.014695  0.000530 -0.031111   
leukemia                    -0.013375         -0.017587  0.001718  0.001014   
lymphoma                    -0.010017         -0.008215  0.008370 -0.002336   
solid_tumor_with_metastasis -0.043380          0.015369  0.004921 -0.014381   
hospital_death              -0.031247         -0.093574 -0.019526  0.000994   

                             pre_icu_los_days    weight  apache_2_diagnosis  \
encounter_id                        -0.000570 -0.004376            0.000113   
patient_id                          -0.004412  0.000275           -0.001539   
hospital_id                         -0.001285  0.026314            0.006806   
age                                  0.049872 -0.127252            0.022914   
bmi                                 -0.001531  0.877339            0.026047   
                                      ...       ...                 ...   
immunosuppression                    0.032695 -0.029642           -0.014445   
leukemia                             0.041853 -0.012755           -0.006915   
lymphoma                             0.013954 -0.005197           -0.006705   
solid_tumor_with_metastasis          0.036581 -0.039255            0.007619   
hospital_death                       0.063316 -0.038362           -0.089862   

                             apache_3j_diagnosis  apache_post_operative  \
encounter_id                           -0.000651               0.001138   
patient_id                              0.004215               0.002260   
hospital_id                             0.031896               0.053985   
age                                    -0.056060               0.059246   
bmi                                    -0.006514               0.015420   
                                         ...                    ...   
immunosuppression                      -0.004731              -0.014487   
leukemia                               -0.006203              -0.013814   
lymphoma                               -0.003000              -0.008186   
solid_tumor_with_metastasis             0.018629               0.012375   
hospital_death                         -0.090715              -0.083674   

                             arf_apache  gcs_eyes_apache  gcs_motor_apache  \
encounter_id                   0.007920         0.003276          0.007664   
patient_id                     0.003980         0.001551          0.002216   
hospital_id                    0.000844        -0.011727         -0.020512   
age                           -0.001684         0.026363          0.025843   
bmi                           -0.005823         0.012927          0.021091   
                                ...              ...               ...   
immunosuppression              0.002247         0.018719          0.017957   
leukemia                       0.011941         0.001230          0.005132   
lymphoma                      -0.003657         0.008648          0.005408   
solid_tumor_with_metastasis   -0.008223         0.012608          0.014312   
hospital_death                 0.027309        -0.260373         -0.282449   

                             gcs_unable_apache  gcs_verbal_apache  \
encounter_id                          0.002400           0.006516   
patient_id                           -0.006673           0.000471   
hospital_id                          -0.017576          -0.002954   
age                                  -0.007754          -0.017829   
bmi                                   0.006613           0.028515   
                                       ...                ...   
immunosuppression                    -0.008968           0.022844   
leukemia                             -0.004221           0.008095   
lymphoma                             -0.004541           0.007984   
solid_tumor_with_metastasis          -0.010249           0.015394   
hospital_death                        0.051774          -0.241044   

                             heart_rate_apache  intubated_apache  map_apache  \
encounter_id                         -0.001718         -0.003172    0.000759   
patient_id                            0.004735          0.001303   -0.002413   
hospital_id                          -0.007649          0.027670    0.001011   
age                                  -0.149495          0.015396   -0.015870   
bmi                                  -0.021118          0.037714    0.055916   
                                       ...               ...         ...   
immunosuppression                     0.054844         -0.008275   -0.015864   
leukemia                              0.019983         -0.003382   -0.016722   
lymphoma                              0.017311         -0.004237   -0.012037   
solid_tumor_with_metastasis           0.040893         -0.011866   -0.008119   
hospital_death                        0.107818          0.173139   -0.040526   

                             resprate_apache  temp_apache  ventilated_apache  \
encounter_id                        0.006498     0.005162          -0.005775   
patient_id                          0.000179     0.002535          -0.004433   
hospital_id                        -0.023262    -0.033490           0.030761   
age                                 0.037328    -0.082265           0.041296   
bmi                                 0.001725     0.038597           0.071014   
                                     ...          ...                ...   
immunosuppression                   0.034752     0.006496          -0.002724   
leukemia                            0.017879     0.003900          -0.002082   
lymphoma                            0.011205    -0.001283          -0.003462   
solid_tumor_with_metastasis         0.012371    -0.000784          -0.016280   
hospital_death                      0.086666    -0.158634           0.228661   

                             d1_diasbp_max  d1_diasbp_min  \
encounter_id                     -0.002547       0.000802   
patient_id                       -0.003287      -0.004180   
hospital_id                      -0.030503       0.020925   
age                              -0.062958      -0.211598   
bmi                               0.054402      -0.007599   
                                   ...            ...   
immunosuppression                -0.011773      -0.013914   
leukemia                         -0.010494      -0.025191   
lymphoma                         -0.007542      -0.011177   
solid_tumor_with_metastasis      -0.018942      -0.002500   
hospital_death                    0.001908      -0.179559   

                             d1_diasbp_noninvasive_max  \
encounter_id                                 -0.002229   
patient_id                                   -0.002946   
hospital_id                                  -0.028594   
age                                          -0.060418   
bmi                                           0.055270   
                                               ...   
immunosuppression                            -0.011594   
leukemia                                     -0.011060   
lymphoma                                     -0.007420   
solid_tumor_with_metastasis                  -0.019470   
hospital_death                                0.002585   

                             d1_diasbp_noninvasive_min  d1_heartrate_max  \
encounter_id                                  0.001202         -0.003891   
patient_id                                   -0.004605          0.001330   
hospital_id                                   0.022370         -0.012901   
age                                          -0.209291         -0.135417   
bmi                                          -0.007641         -0.030567   
                                               ...               ...   
immunosuppression                            -0.014344          0.062603   
leukemia                                     -0.025261          0.023641   
lymphoma                                     -0.011553          0.020198   
solid_tumor_with_metastasis                  -0.002947          0.044415   
hospital_death                               -0.179786          0.162934   

                             d1_heartrate_min  d1_mbp_max  d1_mbp_min  \
encounter_id                        -0.002045   -0.000332   -0.000553   
patient_id                           0.001570   -0.004471   -0.003249   
hospital_id                         -0.005449   -0.027288    0.016304   
age                                 -0.143705    0.006123   -0.131755   
bmi                                  0.015819    0.062054    0.015275   
                                      ...         ...         ...   
immunosuppression                    0.049452   -0.013112   -0.022492   
leukemia                             0.010944   -0.011830   -0.022983   
lymphoma                             0.011790   -0.008453   -0.013356   
solid_tumor_with_metastasis          0.039812   -0.020638   -0.010156   
hospital_death                      -0.003587   -0.016752   -0.195262   

                             d1_mbp_noninvasive_max  d1_mbp_noninvasive_min  \
encounter_id                               0.000734               -0.000472   
patient_id                                -0.004160               -0.003740   
hospital_id                               -0.028030                0.018439   
age                                        0.006198               -0.130206   
bmi                                        0.062675                0.015640   
                                            ...                     ...   
immunosuppression                         -0.011831               -0.023497   
leukemia                                  -0.012855               -0.022929   
lymphoma                                  -0.008277               -0.013603   
solid_tumor_with_metastasis               -0.021109               -0.010721   
hospital_death                            -0.016199               -0.195475   

                             d1_resprate_max  d1_resprate_min  d1_spo2_max  \
encounter_id                        0.006880        -0.000296     0.000041   
patient_id                          0.003159         0.002041    -0.002040   
hospital_id                        -0.042902        -0.005750    -0.001347   
age                                 0.032206         0.034439    -0.038621   
bmi                                 0.003844        -0.010762    -0.068978   
                                     ...              ...          ...   
immunosuppression                   0.036326         0.012808     0.007783   
leukemia                            0.022536         0.015349     0.003627   
lymphoma                            0.014471         0.005653     0.000664   
solid_tumor_with_metastasis         0.018483         0.008982     0.000904   
hospital_death                      0.103093         0.025667    -0.008482   

                             d1_spo2_min  d1_sysbp_max  d1_sysbp_min  \
encounter_id                    0.000512     -0.000484     -0.002455   
patient_id                     -0.002318     -0.000021     -0.006877   
hospital_id                     0.011384     -0.037497      0.014535   
age                            -0.084488      0.107642     -0.060160   
bmi                            -0.024743      0.081232      0.052352   
                                 ...           ...           ...   
immunosuppression              -0.017636     -0.021806     -0.033663   
leukemia                       -0.017838     -0.015333     -0.021674   
lymphoma                       -0.009755     -0.012380     -0.016623   
solid_tumor_with_metastasis    -0.010799     -0.024826     -0.022142   
hospital_death                 -0.210100     -0.027357     -0.210170   

                             d1_sysbp_noninvasive_max  \
encounter_id                                -0.000063   
patient_id                                   0.000188   
hospital_id                                 -0.037839   
age                                          0.107016   
bmi                                          0.081863   
                                              ...   
immunosuppression                           -0.020988   
leukemia                                    -0.015698   
lymphoma                                    -0.011893   
solid_tumor_with_metastasis                 -0.024551   
hospital_death                              -0.026716   

                             d1_sysbp_noninvasive_min  d1_temp_max  \
encounter_id                                -0.002115    -0.004533   
patient_id                                  -0.007695     0.000807   
hospital_id                                  0.015300     0.008949   
age                                         -0.060322    -0.082764   
bmi                                          0.053080     0.023077   
                                              ...          ...   
immunosuppression                           -0.034035     0.017172   
leukemia                                    -0.021868     0.019201   
lymphoma                                    -0.016917     0.001460   
solid_tumor_with_metastasis                 -0.022510    -0.007592   
hospital_death                              -0.209926     0.006293   

                             d1_temp_min  h1_diasbp_max  h1_diasbp_min  \
encounter_id                    0.003035      -0.001529       0.003506   
patient_id                      0.003264      -0.006719      -0.008462   
hospital_id                    -0.039445      -0.002574      -0.010827   
age                            -0.070002      -0.147122      -0.200129   
bmi                             0.035808       0.032378       0.006165   
                                 ...            ...            ...   
immunosuppression               0.000834      -0.017324      -0.019916   
leukemia                       -0.002466      -0.020002      -0.025618   
lymphoma                       -0.001971      -0.011783      -0.012877   
solid_tumor_with_metastasis    -0.002059      -0.013683      -0.008946   
hospital_death                 -0.207239      -0.032064      -0.124924   

                             h1_diasbp_noninvasive_max  \
encounter_id                                  0.000586   
patient_id                                   -0.007044   
hospital_id                                   0.004452   
age                                          -0.136832   
bmi                                           0.032497   
                                               ...   
immunosuppression                            -0.020305   
leukemia                                     -0.022134   
lymphoma                                     -0.013828   
solid_tumor_with_metastasis                  -0.015415   
hospital_death                               -0.035433   

                             h1_diasbp_noninvasive_min  h1_heartrate_max  \
encounter_id                                  0.005045         -0.000167   
patient_id                                   -0.011464          0.000688   
hospital_id                                  -0.002286         -0.015428   
age                                          -0.193331         -0.164788   
bmi                                           0.003592         -0.015905   
                                               ...               ...   
immunosuppression                            -0.022651          0.063718   
leukemia                                     -0.027504          0.022851   
lymphoma                                     -0.014836          0.018565   
solid_tumor_with_metastasis                  -0.011248          0.043911   
hospital_death                               -0.131985          0.113603   

                             h1_heartrate_min  h1_mbp_max  h1_mbp_min  \
encounter_id                        -0.000930   -0.000261    0.002202   
patient_id                           0.001328   -0.005861   -0.008053   
hospital_id                         -0.023639    0.000597   -0.010510   
age                                 -0.165454   -0.059301   -0.108958   
bmi                                 -0.004584    0.039207    0.019265   
                                      ...         ...         ...   
immunosuppression                    0.066032   -0.025966   -0.023227   
leukemia                             0.018047   -0.020900   -0.024695   
lymphoma                             0.017339   -0.012919   -0.012947   
solid_tumor_with_metastasis          0.047358   -0.018336   -0.013279   
hospital_death                       0.087138   -0.061685   -0.141619   

                             h1_mbp_noninvasive_max  h1_mbp_noninvasive_min  \
encounter_id                               0.000654                0.002838   
patient_id                                -0.006467               -0.010379   
hospital_id                                0.000894               -0.004106   
age                                       -0.057717               -0.103732   
bmi                                        0.039372                0.019839   
                                            ...                     ...   
immunosuppression                         -0.027307               -0.026332   
leukemia                                  -0.021942               -0.026438   
lymphoma                                  -0.014282               -0.014626   
solid_tumor_with_metastasis               -0.020559               -0.016480   
hospital_death                            -0.063793               -0.148223   

                             h1_resprate_max  h1_resprate_min  h1_spo2_max  \
encounter_id                        0.002682         0.000351    -0.001577   
patient_id                          0.004802         0.001478     0.001787   
hospital_id                        -0.032078        -0.032271     0.018389   
age                                 0.028071         0.029975    -0.061144   
bmi                                 0.005343        -0.005702    -0.057469   
                                     ...              ...          ...   
immunosuppression                   0.047945         0.036245    -0.007284   
leukemia                            0.024748         0.020652    -0.000130   
lymphoma                            0.019895         0.013573    -0.005631   
solid_tumor_with_metastasis         0.024167         0.018444    -0.000507   
hospital_death                      0.121933         0.110725    -0.047453   

                             h1_spo2_min  h1_sysbp_max  h1_sysbp_min  \
encounter_id                   -0.004182      0.002714      0.000849   
patient_id                      0.000198     -0.003579     -0.006587   
hospital_id                     0.013308     -0.005402     -0.015266   
age                            -0.073128      0.047645     -0.005169   
bmi                            -0.032379      0.059600      0.045603   
                                 ...           ...           ...   
immunosuppression              -0.009274     -0.029809     -0.033947   
leukemia                       -0.003680     -0.019328     -0.022563   
lymphoma                       -0.009324     -0.012771     -0.016002   
solid_tumor_with_metastasis    -0.006963     -0.024314     -0.024316   
hospital_death                 -0.108551     -0.068797     -0.146440   

                             h1_sysbp_noninvasive_max  \
encounter_id                                 0.002974   
patient_id                                  -0.004689   
hospital_id                                 -0.008606   
age                                          0.047147   
bmi                                          0.061362   
                                              ...   
immunosuppression                           -0.030941   
leukemia                                    -0.020026   
lymphoma                                    -0.013881   
solid_tumor_with_metastasis                 -0.025503   
hospital_death                              -0.068093   

                             h1_sysbp_noninvasive_min  d1_glucose_max  \
encounter_id                                 0.001240        0.003672   
patient_id                                  -0.009214       -0.003179   
hospital_id                                 -0.012545       -0.000811   
age                                         -0.003227        0.012538   
bmi                                          0.046477        0.099805   
                                              ...             ...   
immunosuppression                           -0.037182       -0.006710   
leukemia                                    -0.023506       -0.005860   
lymphoma                                    -0.018201       -0.002676   
solid_tumor_with_metastasis                 -0.026903       -0.014106   
hospital_death                              -0.149159        0.081568   

                             d1_glucose_min  d1_potassium_max  \
encounter_id                       0.002068         -0.001755   
patient_id                         0.003235          0.000674   
hospital_id                        0.016819          0.010070   
age                                0.067118          0.061185   
bmi                                0.134702          0.087240   
                                    ...               ...   
immunosuppression                  0.003988         -0.001993   
leukemia                          -0.004478          0.005920   
lymphoma                           0.000885          0.002215   
solid_tumor_with_metastasis        0.011143         -0.001841   
hospital_death                     0.029884          0.112465   

                             d1_potassium_min  apache_4a_hospital_death_prob  \
encounter_id                        -0.000521                      -0.000513   
patient_id                           0.001502                       0.003735   
hospital_id                         -0.012084                      -0.006993   
age                                  0.111465                       0.143167   
bmi                                  0.093457                      -0.033546   
                                      ...                            ...   
immunosuppression                    0.004230                       0.038922   
leukemia                            -0.000508                       0.044126   
lymphoma                             0.004835                       0.018303   
solid_tumor_with_metastasis          0.012820                       0.048300   
hospital_death                       0.025080                       0.311043   

                             apache_4a_icu_death_prob      aids  cirrhosis  \
encounter_id                                 0.000344  0.001907   0.007601   
patient_id                                   0.001625 -0.002426   0.001307   
hospital_id                                 -0.000283 -0.004947   0.002962   
age                                          0.076275 -0.029477  -0.028065   
bmi                                         -0.013796 -0.020434  -0.002377   
                                              ...       ...        ...   
immunosuppression                            0.026268  0.025781  -0.002971   
leukemia                                     0.031966 -0.002471  -0.005373   
lymphoma                                     0.011287  0.021529   0.001516   
solid_tumor_with_metastasis                  0.028090 -0.001611  -0.005890   
hospital_death                               0.283913  0.004403   0.039453   

                             diabetes_mellitus  hepatic_failure  \
encounter_id                          0.003402        -0.000972   
patient_id                            0.000434        -0.001667   
hospital_id                           0.011978         0.001362   
age                                   0.077908        -0.020061   
bmi                                   0.172943        -0.001855   
                                       ...              ...   
immunosuppression                    -0.002502         0.003084   
leukemia                              0.002890        -0.001567   
lymphoma                             -0.002326         0.001689   
solid_tumor_with_metastasis          -0.013122         0.007240   
hospital_death                       -0.015784         0.038864   

                             immunosuppression  leukemia  lymphoma  \
encounter_id                         -0.001707 -0.002172 -0.001176   
patient_id                            0.000548  0.000844  0.002333   
hospital_id                           0.000146 -0.002992  0.002588   
age                                   0.025007  0.030310  0.023335   
bmi                                  -0.031144 -0.013375 -0.010017   
                                       ...       ...       ...   
immunosuppression                     1.000000  0.134934  0.103201   
leukemia                              0.134934  1.000000  0.031380   
lymphoma                              0.103201  0.031380  1.000000   
solid_tumor_with_metastasis           0.269653  0.006210  0.014749   
hospital_death                        0.043973  0.029788  0.018722   

                             solid_tumor_with_metastasis  hospital_death  
encounter_id                                   -0.004646       -0.005243  
patient_id                                     -0.003065        0.004877  
hospital_id                                    -0.004730       -0.001255  
age                                             0.025924        0.111017  
bmi                                            -0.043380       -0.031247  
                                                 ...             ...  
immunosuppression                               0.269653        0.043973  
leukemia                                        0.006210        0.029788  
lymphoma                                        0.014749        0.018722  
solid_tumor_with_metastasis                     1.000000        0.051105  
hospital_death                                  0.051105        1.000000  

[77 rows x 77 columns]

#Характеристики категориальных признаков

data.describe(include = ['object'])
Out  [3]: 
        ethnicity gender      icu_admit_source icu_stay_type      icu_type  \
count       90318  91688                 91601         91713         91713   
unique          6      2                     5             3             8   
top     Caucasian      M  Accident & Emergency         admit  Med-Surg ICU   
freq        70684  49469                 54060         86183         50586   

       apache_3j_bodysystem apache_2_bodysystem  
count                 90051               90051  
unique                   11                  10  
top          Cardiovascular      Cardiovascular  
freq                  29999               38816  

'''

Обрабатываем пропущенные значения

'''

#Считаем количество пропущенных значений

data.isnull().sum()
Out  [4]: 
encounter_id                        0
patient_id                          0
hospital_id                         0
age                              4228
bmi                              3429
elective_surgery                    0
ethnicity                        1395
gender                             25
height                           1334
icu_admit_source                  112
icu_id                              0
icu_stay_type                       0
icu_type                            0
pre_icu_los_days                    0
weight                           2720
apache_2_diagnosis               1662
apache_3j_diagnosis              1101
apache_post_operative               0
arf_apache                        715
gcs_eyes_apache                  1901
gcs_motor_apache                 1901
gcs_unable_apache                1037
gcs_verbal_apache                1901
heart_rate_apache                 878
intubated_apache                  715
map_apache                        994
resprate_apache                  1234
temp_apache                      4108
ventilated_apache                 715
d1_diasbp_max                     165
d1_diasbp_min                     165
d1_diasbp_noninvasive_max        1040
d1_diasbp_noninvasive_min        1040
d1_heartrate_max                  145
d1_heartrate_min                  145
d1_mbp_max                        220
d1_mbp_min                        220
d1_mbp_noninvasive_max           1479
d1_mbp_noninvasive_min           1479
d1_resprate_max                   385
d1_resprate_min                   385
d1_spo2_max                       333
d1_spo2_min                       333
d1_sysbp_max                      159
d1_sysbp_min                      159
d1_sysbp_noninvasive_max         1027
d1_sysbp_noninvasive_min         1027
d1_temp_max                      2324
d1_temp_min                      2324
h1_diasbp_max                    3619
h1_diasbp_min                    3619
h1_diasbp_noninvasive_max        7350
h1_diasbp_noninvasive_min        7350
h1_heartrate_max                 2790
h1_heartrate_min                 2790
h1_mbp_max                       4639
h1_mbp_min                       4639
h1_mbp_noninvasive_max           9084
h1_mbp_noninvasive_min           9084
h1_resprate_max                  4357
h1_resprate_min                  4357
h1_spo2_max                      4185
h1_spo2_min                      4185
h1_sysbp_max                     3611
h1_sysbp_min                     3611
h1_sysbp_noninvasive_max         7341
h1_sysbp_noninvasive_min         7341
d1_glucose_max                   5807
d1_glucose_min                   5807
d1_potassium_max                 9585
d1_potassium_min                 9585
apache_4a_hospital_death_prob    7947
apache_4a_icu_death_prob         7947
aids                              715
cirrhosis                         715
diabetes_mellitus                 715
hepatic_failure                   715
immunosuppression                 715
leukemia                          715
lymphoma                          715
solid_tumor_with_metastasis       715
apache_3j_bodysystem             1662
apache_2_bodysystem              1662
hospital_death                      0
dtype: int64

#Удаляем строки, в которых заполнено меньше 90% данных

data = data.dropna(axis = 0, thresh = data.shape[1] * 0.9)
data = data.reset_index()
data = data.drop('index', axis = 1)
data
Out  [5]: 
       encounter_id  patient_id  ...  apache_2_bodysystem  hospital_death
0             66154       25312  ...       Cardiovascular               0
1            114252       59342  ...          Respiratory               0
2            119783       50777  ...            Metabolic               0
3             79267       46918  ...       Cardiovascular               0
4             33181       74489  ...           Neurologic               0
            ...         ...  ...                  ...             ...
86021        118430       83320  ...            Metabolic               0
86022        127138       59223  ...       Cardiovascular               0
86023         91592       78108  ...       Cardiovascular               0
86024         66119       13486  ...       Cardiovascular               0
86025          1671       53612  ...     Gastrointestinal               0

[86026 rows x 84 columns]

#Берём медиану для числовых значений

data_num = data.median(axis = 0, numeric_only = True)
data_num
Out  [6]: 
encounter_id                   65736.500000
patient_id                     65452.500000
hospital_id                      112.000000
age                               65.000000
bmi                               27.616817
    
immunosuppression                  0.000000
leukemia                           0.000000
lymphoma                           0.000000
solid_tumor_with_metastasis        0.000000
hospital_death                     0.000000
Length: 77, dtype: float64

#Берём популярные значения для категориальных признаков

data_categ = data.describe(include = ['object']).loc['top']
data_categ
Out  [7]: 
ethnicity                          Caucasian
gender                                     M
icu_admit_source        Accident & Emergency
icu_stay_type                          admit
icu_type                        Med-Surg ICU
apache_3j_bodysystem          Cardiovascular
apache_2_bodysystem           Cardiovascular
Name: top, dtype: object

#Заполняем этими данными пустые места

data = data.fillna(data_num, axis = 0)
data = data.fillna(data_categ, axis = 0)

#Проверяем, что теперь пропущенных значений не осталось

data.isnull().sum()
Out  [8]: 
encounter_id                     0
patient_id                       0
hospital_id                      0
age                              0
bmi                              0
elective_surgery                 0
ethnicity                        0
gender                           0
height                           0
icu_admit_source                 0
icu_id                           0
icu_stay_type                    0
icu_type                         0
pre_icu_los_days                 0
weight                           0
apache_2_diagnosis               0
apache_3j_diagnosis              0
apache_post_operative            0
arf_apache                       0
gcs_eyes_apache                  0
gcs_motor_apache                 0
gcs_unable_apache                0
gcs_verbal_apache                0
heart_rate_apache                0
intubated_apache                 0
map_apache                       0
resprate_apache                  0
temp_apache                      0
ventilated_apache                0
d1_diasbp_max                    0
d1_diasbp_min                    0
d1_diasbp_noninvasive_max        0
d1_diasbp_noninvasive_min        0
d1_heartrate_max                 0
d1_heartrate_min                 0
d1_mbp_max                       0
d1_mbp_min                       0
d1_mbp_noninvasive_max           0
d1_mbp_noninvasive_min           0
d1_resprate_max                  0
d1_resprate_min                  0
d1_spo2_max                      0
d1_spo2_min                      0
d1_sysbp_max                     0
d1_sysbp_min                     0
d1_sysbp_noninvasive_max         0
d1_sysbp_noninvasive_min         0
d1_temp_max                      0
d1_temp_min                      0
h1_diasbp_max                    0
h1_diasbp_min                    0
h1_diasbp_noninvasive_max        0
h1_diasbp_noninvasive_min        0
h1_heartrate_max                 0
h1_heartrate_min                 0
h1_mbp_max                       0
h1_mbp_min                       0
h1_mbp_noninvasive_max           0
h1_mbp_noninvasive_min           0
h1_resprate_max                  0
h1_resprate_min                  0
h1_spo2_max                      0
h1_spo2_min                      0
h1_sysbp_max                     0
h1_sysbp_min                     0
h1_sysbp_noninvasive_max         0
h1_sysbp_noninvasive_min         0
d1_glucose_max                   0
d1_glucose_min                   0
d1_potassium_max                 0
d1_potassium_min                 0
apache_4a_hospital_death_prob    0
apache_4a_icu_death_prob         0
aids                             0
cirrhosis                        0
diabetes_mellitus                0
hepatic_failure                  0
immunosuppression                0
leukemia                         0
lymphoma                         0
solid_tumor_with_metastasis      0
apache_3j_bodysystem             0
apache_2_bodysystem              0
hospital_death                   0
dtype: int64

'''

Обрабатываем категориальные признаки

'''

#Разделяем столбцы на категориальные и числовые

categ_col = [c for c in data.columns if data[c].dtype.name == 'object']
num_col = [c for c in data.columns if data[c].dtype.name != 'object']

#Разделяем категориальные признаки на бинарные и небинарные

binary_col = [c for c in categ_col if data[c].unique().size == 2]
nonbinary_col = [c for c in categ_col if data[c].unique().size > 2]

binary_col
Out  [9]: ['gender']
nonbinary_col
Out  [10]: ['ethnicity', 'icu_admit_source', 'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']

#Работаем с бинарным признаком. Заменяем текстовые значения нулями и единичками

data['gender'].unique()
Out  [11]: array(['M', 'F'], dtype=object)

data = data.replace({'gender': {'M': 0, 'F': 1}})
data['gender'].describe()
Out  [12]: 
count    86026.000000
mean         0.460919
std          0.498473
min          0.000000
25%          0.000000
50%          0.000000
75%          1.000000
max          1.000000
Name: gender, dtype: float64

#Работаем с небинарными признаками. Разделяем каждый признак с s значениями на s признаков со значениями 0 и 1

data_nonbin = pd.get_dummies(data[nonbinary_col])
data_nonbin.describe()
Out  [13]: 
       ethnicity_African American  ...  apache_2_bodysystem_Undefined diagnoses
count                86026.000000  ...                             86026.000000
mean                     0.103643  ...                                 0.038698
std                      0.304799  ...                                 0.192874
min                      0.000000  ...                                 0.000000
25%                      0.000000  ...                                 0.000000
50%                      0.000000  ...                                 0.000000
75%                      0.000000  ...                                 0.000000
max                      1.000000  ...                                 1.000000

[8 rows x 43 columns]

'''

Нормализация


'''

#Текущие числовые характеристики. Убрали столбец 'hospital_death', так как это класс, а не признак

data_num = data[num_col]
data_num = data_num.drop('hospital_death', axis = 1)
data_num.describe()
Out  [14]: 
        encounter_id     patient_id  ...      lymphoma  solid_tumor_with_metastasis
count   86026.000000   86026.000000  ...  86026.000000                 86026.000000
mean    65612.543115   65562.782356  ...      0.004196                     0.021040
std     37798.685560   37786.063465  ...      0.064644                     0.143519
min         1.000000       1.000000  ...      0.000000                     0.000000
25%     32821.750000   32907.250000  ...      0.000000                     0.000000
50%     65736.500000   65452.500000  ...      0.000000                     0.000000
75%     98301.750000   98291.000000  ...      0.000000                     0.000000
max    131051.000000  131051.000000  ...      1.000000                     1.000000

[8 rows x 76 columns]

#Нормализуем числовые данные. Приводим их к в виду, когда среднее нулевое, а стандартное отклонение единичное

data_num = (data_num - data_num.mean()) / data_num.std()
data_num.describe()
Out  [15]: 
       encounter_id    patient_id  ...      lymphoma  solid_tumor_with_metastasis
count  8.602600e+04  8.602600e+04  ...  8.602600e+04                 8.602600e+04
mean   1.610627e-16  4.935128e-17  ...  1.321540e-18                 2.180542e-17
std    1.000000e+00  1.000000e+00  ...  1.000000e+00                 1.000000e+00
min   -1.735815e+00 -1.735078e+00  ... -6.491564e-02                -1.466019e-01
25%   -8.675115e-01 -8.642216e-01  ... -6.491564e-02                -1.466019e-01
50%    3.279397e-03 -2.918599e-03  ... -6.491564e-02                -1.466019e-01
75%    8.648239e-01  8.661452e-01  ... -6.491564e-02                -1.466019e-01
max    1.731236e+00  1.733132e+00  ...  1.540443e+01                 6.821117e+00

[8 rows x 76 columns]

'''

Соединяем все результаты в одну таблицу

'''

data = pd.concat((data_num, data_nonbin, data[binary_col], data['hospital_death']), axis = 1)
data.describe()
Out  [16]: 
       encounter_id    patient_id  ...        gender  hospital_death
count  8.602600e+04  8.602600e+04  ...  86026.000000    86026.000000
mean   1.610627e-16  4.935128e-17  ...      0.460919        0.086358
std    1.000000e+00  1.000000e+00  ...      0.498473        0.280893
min   -1.735815e+00 -1.735078e+00  ...      0.000000        0.000000
25%   -8.675115e-01 -8.642216e-01  ...      0.000000        0.000000
50%    3.279397e-03 -2.918599e-03  ...      0.000000        0.000000
75%    8.648239e-01  8.661452e-01  ...      1.000000        0.000000
max    1.731236e+00  1.733132e+00  ...      1.000000        1.000000

[8 rows x 121 columns]

'''

Подготовливаем данные к непосредственному применению

'''

#Разделили признаки и результат
X = data.drop('hospital_death', axis = 1)
y = data['hospital_death']

#Разделили данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train.shape[0]
Out  [17]: 64519
X_test.shape[0]
Out  [18]: 21507

'''

Метод ближайших соседей

'''

#Пробный запуск

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)

#Погрешность на обучающей выборке
err_test = 1 - knn.score(X_test, y_test)
err_test
Out  [19]: 0.07816059887478499

#Погрешность на тренировочной выборке
err_train = 1 - knn.score(X_train, y_train)
err_train
Out  [20]: 0.0757761279623057

'''
Вывод: метод k ближайших соседей (k = 10) дал довольно хороший результат. Погрешности на тренировочной и обучающей выборках
не сильно отличаются и сравнительно малы, значит переобучения и недообучения не произошло. Задача классификации решается достаточно точно
'''

'''
Что эквивалентно y_test_predict = knn.predict(X_test), err = 1 - np.mean(y_test == y_test_predict)
'''

#Ищем лучшее количество соседей из 5 вариантов, представленных ниже

nnb = [4, 8, 10, 15, 19]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid = {'n_neighbors': nnb})
grid.fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print(best_cv_err, best_n_neighbors)
Out  [21]: 0.07892249234944393 15

knn = KNeighborsClassifier(n_neighbors = best_n_neighbors)
knn.fit(X_train, y_train)

err_train = 1 - knn.score(X_train, y_train)
err_test = 1 - knn.score(X_test, y_test)
print(err_train, err_test)
Out  [22]: 0.07562113485949873 0.07946250058120607

'''

Random Forest

'''

rf = ensemble.RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train, y_train)

err_train = 1 - rf.score(X_train, y_train)
err_test = 1 - rf.score(X_test, y_test)
print(err_train, err_test)
Out  [23]: 0.0 0.07165109034267914

'''

AdaBoost

'''

ada = ensemble.AdaBoostClassifier(n_estimators = 1000)
ada.fit(X_train, y_train)

err_train = 1 - ada.score(X_train, y_train)
err_test = 1 - ada.score(X_test, y_test)
print(err_train, err_test)
Out  [24]: 0.06884793626683616 0.07146510438461895

'''

Extremely Randomized Trees

'''

ert = ensemble.ExtraTreesClassifier(n_estimators = 1000)
ert.fit(X_train, y_train)

err_train = 1 - ert.score(X_train, y_train)
err_test = 1 - ert.score(X_test, y_test)
print(err_train, err_test)
Out  [25]: 0.0 0.07183707630073932

'''


Общие выводы: Удалось построить модель, которая с достаточной степенью точности способна спрогнозировать летальность госпитализированного
в зависимости от причин госпитализации.
Лучший результат дал метод AdaBoost



'''

'''

Чистый код

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble

fileway = 'C:\\Users\\HOME\\Desktop\\dataset.csv'
data = pd.read_csv(fileway, sep = ',')

#Удаляем строки, в которых заполнено меньше 90% данных
data = data.dropna(axis = 0, thresh = data.shape[1] * 0.9)
data = data.reset_index()
data = data.drop('index', axis = 1)

#Берём медиану для числовых значений
data_num = data.median(axis = 0, numeric_only = True)
#Берём популярные значения для категориальных признаков
data_categ = data.describe(include = ['object']).loc['top']
#Заполняем этими данными пустые места
data = data.fillna(data_num, axis = 0)
data = data.fillna(data_categ, axis = 0)

#Разделяем столбцы на категориальные и числовые
categ_col = [c for c in data.columns if data[c].dtype.name == 'object']
num_col = [c for c in data.columns if data[c].dtype.name != 'object']

#Разделяем категориальные признаки на бинарные и небинарные
binary_col = [c for c in categ_col if data[c].unique().size == 2]
nonbinary_col = [c for c in categ_col if data[c].unique().size > 2]

#Работаем с бинарным признаком. Заменяем текстовые значения нулями и единичками
data = data.replace({'gender': {'M': 0, 'F': 1}})

#Работаем с небинарными признаками. Разделяем каждый признак с s значениями на s признаков со значениями 0 и 1
data_nonbin = pd.get_dummies(data[nonbinary_col])

#Текущие числовые характеристики. Убрали столбец 'hospital_death', так как это класс, а не признак
data_num = data[num_col]
data_num = data_num.drop('hospital_death', axis = 1)

#Нормализуем числовые данные. Приводим их к в виду, когда среднее нулевое, а стандартное отклонение единичное
data_num = (data_num - data_num.mean()) / data_num.std()

#Соединяем все результаты в одну таблицу
data = pd.concat((data_num, data_nonbin, data[binary_col], data['hospital_death']), axis = 1)

#Разделили признаки и результат
X = data.drop('hospital_death', axis = 1)
y = data['hospital_death']

#Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

breakpoint()

'''
#Ищем лучшее количество соседей из 5 вариантов, представленных ниже

nnb = [4, 8, 10, 15, 19]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid = {'n_neighbors': nnb})
grid.fit(X_train, y_train)

best_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print(best_err, best_n_neighbors)

#Выбираем лучший и смотрим результаты

knn = KNeighborsClassifier(n_neighbors = best_n_neighbors)
knn.fit(X_train, y_train)

err_train = 1 - knn.score(X_train, y_train)
err_test = 1 - knn.score(X_test, y_test)
print(err_train, err_test)
'''

'''
#Random Forest
rf = ensemble.RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train, y_train)

err_train = 1 - rf.score(X_train, y_train)
err_test = 1 - rf.score(X_test, y_test)
print(err_train, err_test)
'''

'''
#Ada Boost
ada = ensemble.AdaBoostClassifier(n_estimators = 1000)
ada.fit(X_train, y_train)

err_train = 1 - ada.score(X_train, y_train)
err_test = 1 - ada.score(X_test, y_test)
print(err_train, err_test)
'''

'''
#Extremely Randomized Trees
ert = ensemble.ExtraTreesClassifier(n_estimators = 100)
ert.fit(X_train, y_train)

err_train = 1 - ert.score(X_train, y_train)
err_test = 1 - ert.score(X_test, y_test)
print(err_train, err_test)
'''
