# Predicting O-Level Mathematics Scores

Regression pipeline that predicts a student's `final_test` score from their
school records, so the school can identify weaker students before the
examination.

The best model reaches a test RMSE of **5.21 marks** and R2 of **0.87**,
against 9.25 for a linear model and 14.20 for guessing the average.

## Folder structure

```
.
├── data/
│   ├── regression_bonus_practice_data.csv   raw data, never modified
│   └── cleaned.csv                          output of the cleaning step
├── src/
│   ├── config.yaml                          all parameters live here
│   ├── data_preparation.py                  cleaning and feature engineering
│   └── model_training.py                    split, preprocessing, models, evaluation
├── results/
│   ├── model_comparison.csv                 metrics for every model
│   ├── feature_importances.csv              importances for the chosen model
│   ├── grid_search_results.csv              every hyperparameter combination tried
│   ├── error_profile.csv                    how far off the chosen model is, in marks
│   └── best_params.csv                      chosen hyperparameters
├── eda.ipynb                                exploration and the reasoning behind the cleaning
├── requirements.txt
└── run.sh
```

## Running it

```bash
pip install -r requirements.txt
./run.sh
```

Or run the two steps separately, from the project root:

```bash
python src/data_preparation.py    # writes data/cleaned.csv
python src/model_training.py      # writes results/
```

Training takes about a minute. Parameters are in `src/config.yaml`, so the
paths, the split ratio, the imputation strategy and the hyperparameter grids
can all be changed without touching the code.

## Pipeline

```
raw csv
   -> clean          drop unusable columns, fix typos, standardise labels, dedupe
   -> engineer       class_size, sleep_hours
   -> split 80/20    before anything is fitted
   -> preprocess     impute, one hot encode, scale (fitted on train folds only)
   -> train          6 models, 5 fold cross validation, grid search
   -> evaluate       test set scored once, at the end
```

The split comes before any fitted transformation. Everything that learns a
value from the data sits inside a scikit-learn `Pipeline`, so cross validation
refits it on each fold and no test information reaches the training data.

This is why `data/cleaned.csv` still contains missing `attendance_rate`
values. Imputing them during cleaning would have computed the median over the
test rows as well.

## EDA findings

The findings that shaped the pipeline:

**`class_size` is the strongest predictor** (correlation -0.50), and it does
not exist in the raw data. It was built from `n_male + n_female`, neither of
which is useful alone (-0.15 and -0.17).

**`hours_per_week` is not linearly related to the score.** The raw correlation
is -0.15, which reads as "studying more lowers your score". Averaging the score
at each individual value shows three flat stretches with two jumps between them
(45-47 up to 5 hours, 70-74 from 5 to 10, 60-64 from 11 to 20), which is where
these three bands come from:

| hours/week | mean score | students |
|---|---|---|
| 0-4 | 46.4 | 578 |
| 5-10 | 72.2 | 8190 |
| 11-20 | 62.1 | 5786 |

An upside down U, best around 5-10 hours. Inside each band the correlation is
flat or slightly positive (+0.06, -0.01, +0.14). The overall figure only comes
out negative because the 11-20 group sits lower than the 5-10 group, not
because extra hours pull a score down.

This is not attendance or family size in disguise. Comparing only students who
have similar attendance and the same number of siblings, the 5-10 group still
scores highest in all nine groups checked.

**Missing `CCA` means no CCA, not unknown.** The column was entered in two
styles. The uppercase rows offer four options including `NONE` and are never
blank; the title case rows offer three options and are blank 24.9% of the
time. Both split roughly four ways, and the blanks score 76.8 against `NONE`
at 76.7. The blanks are the no-CCA students from the second data source.
Those students score about 13 marks above CCA participants, so the category is
kept rather than imputed or dropped.

**`number_of_siblings` does not drop steadily.** 0 siblings averages 76.0, 1
averages 61.9, 2 averages 63.9. One sibling scores lower than two, so the real
split is having no siblings against having any.

**`age` tells us nothing** once the typos are fixed. Age 15 averages 67.15 and
age 16 averages 67.23. Splitting the students by tuition, direct admission,
learning style and sibling count, the difference between the two ages is under
one mark every time. It is still given to the model, which ranks it second
from last out of 15 features.

**The straightforward columns.** `direct_admission`, `learning_style`,
`tuition` and `CCA` each split students cleanly into a better and a worse
group, and each is a simple category a linear model can handle without
trouble:

| column | lower group | higher group |
|---|---|---|
| `direct_admission` | No 65.0 | Yes 72.5 |
| `learning_style` | Auditory 63.9 | Visual 71.6 |
| `tuition` | No 62.9 | Yes 70.5 |
| `CCA` | Arts/Clubs/Sports about 64 | None 76.8 |

**`attendance_rate` is lopsided**, which decides how its gaps get filled. Most
students sit in the 90s, with a thin tail running down to 40. That tail drags
the mean below the middle of the data (mean 93.3, median 95.0), so the median
is the safer value to fill with. It also matters that the low attendance
students are the ones the school wants flagged, so distorting them would
defeat the point. Correlation with the score is 0.35.

**`sleep_hours` works but is lopsided too.** More sleep does mean a better
score, steadily: 43.8 marks at 4 hours rising to 68.4 at 8. But 92% of
students are at 8 hours, so the feature really only separates out a small
group of tired students. Kept, but not leaned on.

**`class_size` and `sleep_hours` are the only created features.** Everything
else is a raw column, kept or dropped as listed below.

Three columns do not move steadily up or down with the score:

| feature | what it does |
|---|---|
| `class_size` | mostly falls as classes get bigger, but size 17 averages 82.6, higher than any smaller class |
| `hours_per_week` | rises to a peak at 5-10 hours, then falls |
| `number_of_siblings` | 76.0 at none, down to 61.9 at one, back up to 63.9 at two |

A linear model draws one straight line through each feature, so it gets all
three of these wrong. A tree can cut a feature into ranges and treat each range
differently, so it can follow these shapes. That is a reason to expect the tree
models to do better here.

### Every column, and what happened to it

| column | decision | reason |
|---|---|---|
| `index` | dropped | a row number, carries nothing |
| `number_of_siblings` | kept | none 76.0, one 61.9, two 63.9 |
| `direct_admission` | kept | Yes 72.5, No 65.0 |
| `CCA` | kept, labels standardised | blanks merged into `NONE`, which scores 76.8 against about 64 |
| `learning_style` | kept | Visual 71.6, Auditory 63.9 |
| `student_id` | used, then dropped | finds the duplicate students, predicts nothing |
| `gender` | dropped | flat against the score |
| `tuition` | kept, labels standardised | `Y`/`N` merged into `Yes`/`No`. Yes 70.5, No 62.9 |
| `final_test` | target | 495 rows with no value are dropped |
| `n_male` | used, then dropped | folded into `class_size` |
| `n_female` | used, then dropped | folded into `class_size` |
| `age` | kept | typos fixed. 15 and 16 score the same, so it adds nothing. left in anyway, and the trained model does rank it last |
| `hours_per_week` | kept | peaks at 5-10 hours instead of rising or falling steadily. the trained model relies on it more than any column except `class_size` and `number_of_siblings` |
| `attendance_rate` | kept | correlation 0.35, gaps filled with the median after the split |
| `sleep_time` | used, then dropped | folded into `sleep_hours` |
| `wake_time` | used, then dropped | folded into `sleep_hours` |
| `mode_of_transport` | dropped | flat against the score |
| `bag_color` | dropped | flat against the score, and only ever differed between duplicate rows |
| `class_size` | created | `n_male + n_female`. strongest feature at -0.50 |
| `sleep_hours` | created | `wake_time - sleep_time`. correlation 0.33 |

## Assumptions

Each of these is a judgement call, listed so it can be challenged:

1. **Students sitting O-levels are 15 or 16.** Ages of -5 and -4 (5 rows) are
   unrecoverable and those rows are dropped. Ages of 5 and 6 (446 rows) are
   read as 15 and 16 with a missing leading digit. The pipeline raises an error
   if any other age survives cleaning.
2. **Blank `CCA` means the student has no CCA**, per the entry-style evidence
   above. It is merged into `NONE`.
3. **`Y`/`N` and `Yes`/`No` in `tuition` are the same values**, differently
   typed.
4. **Rows with no `final_test` cannot be used** (495 rows). They are the target.
5. **Repeated `student_id`s are the same student recorded twice.** After
   dropping exact duplicates, the remaining repeats differ only in `bag_color`
   and `attendance_rate`. One row per student is kept, preferring the row with
   a recorded `attendance_rate`.
6. **`bag_color`, `gender` and `mode_of_transport` carry no signal** and are
   dropped. Their boxplots against `final_test` are flat.
7. **Missing `attendance_rate` is missing at random.** The missing rate is
   4.4-5.0% across every hours band and every sibling count, with no pattern
   found. It is imputed with the median rather than the mean, because the tail
   down to 40 drags the mean below the middle of the data. A missing indicator
   column is added so the model can still tell a filled value from a recorded
   one.

## Features

| feature | type | treatment |
|---|---|---|
| `number_of_siblings` | numeric | scaled |
| `age` | numeric | scaled |
| `hours_per_week` | numeric | scaled |
| `attendance_rate` | numeric | median imputed + indicator, scaled |
| `class_size` | numeric, engineered | `n_male + n_female`, scaled |
| `sleep_hours` | numeric, engineered | `wake_time - sleep_time` mod 24, scaled |
| `direct_admission` | categorical | one hot |
| `CCA` | categorical | one hot |
| `learning_style` | categorical | one hot |
| `tuition` | categorical | one hot |

`student_id` is used to deduplicate and is then dropped. It identifies a
student, it does not predict anything.

Scaling only matters to the linear models, but applying it to every model
keeps a single preprocessor and makes the comparison fair.

## Models

Four models plus a reference line, all through the same preprocessing pipeline:

| model | what it does |
|---|---|
| Baseline (mean) | predicts the same average for every student. Not a model, just the score everything else has to beat |
| Linear Regression | fits one straight line through each feature. Coefficients can be read directly |
| Decision Tree | cuts each feature into ranges and predicts an average inside each range |
| Random Forest | builds many trees on random subsets of the data and averages them |
| Gradient Boosting | builds trees one after another, each one correcting the errors of the last |

Selection used 5-fold cross validation on the training set only. The three tree
models were tuned with `GridSearchCV` over the grids in `config.yaml`, and the
cross validation runs after tuning, so both columns in the table below describe
the same estimator. The test set was scored once, after every decision had been
made.

## Results

Cross validation is on the training set, test is the held-out 20%.

| model | CV MAE | CV MSE | CV RMSE | CV R2 | | Test MAE | Test MSE | Test RMSE | Test R2 |
|---|---|---|---|---|:-:|---|---|---|---|
| **Random Forest** | **3.741** | **28.78** | **5.363** | **0.851** | │ | **3.660** | **27.11** | **5.207** | **0.866** |
| Gradient Boosting | 3.864 | 29.86 | 5.462 | 0.846 | │ | 3.832 | 28.67 | 5.355 | 0.858 |
| Decision Tree | 3.931 | 32.60 | 5.708 | 0.832 | │ | 3.793 | 29.69 | 5.449 | 0.853 |
| Linear Regression | 7.229 | 81.81 | 9.044 | 0.577 | │ | 7.420 | 85.56 | 9.250 | 0.576 |
| Baseline (mean) | 11.586 | 193.79 | 13.920 | -0.001 | │ | 11.902 | 201.66 | 14.201 | -0.000 |

Chosen hyperparameters: Decision Tree `max_depth=10, min_samples_leaf=5`,
Random Forest `max_depth=10, min_samples_leaf=1, max_features=1.0,
n_estimators=200`, Gradient Boosting `learning_rate=0.05, max_depth=None,
max_iter=300`. The two trees differ on `min_samples_leaf` deliberately, see One
tuning decision below.

Each of the three tree models was given a grid of exactly 12 combinations, so
none of them wins by having been searched harder than the others. All 36 are in
`results/grid_search_results.csv`.

### Reading the results

The baseline ignores every column and predicts the same mark for everyone:
67.24, the training average. That is off by 14.20 RMSE, which matches how spread
out the marks are to begin with. So 14.20 is the bar, and every model clears it.

Linear Regression gets about a third of the way down, to 9.25.

The real gain comes from cutting features into ranges. A single Decision Tree
drops CV RMSE from 9.04 to 5.71, most of the remaining distance on its own. This
is what the EDA pointed at: once a model can treat different ranges of
`class_size`, `number_of_siblings` and `hours_per_week` differently, the error
falls away.

The ensembles add much less on top, 5.363 for Random Forest and 5.462 for
Gradient Boosting. Worth having, but small next to what the tree structure alone
bought. The signal here looks simple enough that averaging trees already
captures it, leaving boosting little to correct.

Test scores come out slightly better than cross validation for every model,
which is expected: the final model trains on all 11,643 training rows, while
each cross validation fold only gets four fifths of them.

## Chosen model

**Random Forest.** Best on every metric, in cross validation and on test alike:
5.363 and 5.207 RMSE, against 5.462 / 5.355 for Gradient Boosting and 5.708 /
5.449 for the Decision Tree. Fold-to-fold spread in RMSE is 0.136, between the
Decision Tree's 0.128 and Gradient Boosting's 0.165, so the win does not come
with any extra instability.

If explainability mattered more than accuracy, the Decision Tree would be the
pick: 0.24 RMSE worse, but it prints as a single picture. A prediction of 45
traces down the branches, class size above 24, then one or more siblings, then
attendance below 90, and a teacher can judge whether that path makes sense.
Random Forest averages 200 trees, so the number comes out but the reason does
not. A teacher told to pull a student aside will ask why. Random Forest is
chosen because the brief asks for the best estimate of the score, but the
Decision Tree is the one to reach for if the model has to justify itself.

### How far off is it, in plain terms

For the 2,911 test students, written to `results/error_profile.csv`:

| | |
|---|---|
| half the students are missed by less than | 2.6 marks |
| the average miss | 3.7 marks |
| 9 in 10 missed by less than | 8.5 marks |
| 99 in 100 missed by less than | 18.0 marks |
| worst single miss | 31.3 marks |
| predicted within 5 marks | 76.5% of students |
| predicted within 10 marks | 93.2% of students |

The typical student is predicted within about 3 marks. Three quarters land
within 5, roughly one O-level grade band, so most students are placed in the
right band or the one next to it. The tail is the honest limit: 1 student in
100 is out by 18 marks or more, and the worst case is 31. That is why this is a
tool for flagging students to look at, not for deciding anything on its own.

### One tuning decision worth stating

The Decision Tree grid originally allowed `min_samples_leaf=1`, and it won on
score, 5.632 CV RMSE against 5.708 for a minimum of 5. It was rejected anyway:
that tree had 684 leaves, 192 of them holding a single student, so those
predictions were one classmate's score copied across. It affects about 5.6% of
students, which is why cross validation barely noticed, but it is the wrong
shape for the model whose value is being explainable.

The grid now starts at 5, costing 0.08 CV RMSE for predictions that always rest
on at least five students. Random Forest keeps 1, because averaging 200 trees
already washes out any single tree's tiny leaves. A grid search cannot make
this call, since on score alone the answer is 1.

### What drives the prediction

A trained Random Forest reports how much each feature contributed. The numbers
add to 1, so 0.393 means that feature accounts for about 39% of what the model
does. Full list in `results/feature_importances.csv`:

| feature | importance |
|---|---|
| `class_size` | 0.393 |
| `number_of_siblings` | 0.268 |
| `hours_per_week` | 0.112 |
| `learning_style` | 0.082 |
| `attendance_rate` | 0.057 |
| `tuition` | 0.035 |
| `direct_admission` | 0.031 |
| `CCA` | 0.020 |
| `sleep_hours` | 0.002 |
| `age` | 0.002 |

The one hot columns are added back together here, so each row is one original
column and the ten add up to 1. The saved file lists them unmerged.

The top three are exactly the three columns that do not move in a straight line
with the score, one of which had to be created first. `age` comes last, as the
EDA predicted.

`sleep_hours` sitting at 0.002 looks like it contradicts the EDA, where
sleeping under 7 hours capped scores near 50. It does not. `sleep_hours`
correlates 0.886 with `attendance_rate`, and the two barely overlap in range:
students on 4 hours of sleep have attendance between 40 and 63, students on 8
hours between 76 and 100. A student who sleeps little and attends often
essentially does not exist in this data.

Holding attendance still, sleep stops mattering. Inside the lowest attendance
band, mean scores run 43.8, 45.1, 45.7, 45.5, 46.1 across 4 to 8 hours of
sleep, which is flat. So the forest leans on `attendance_rate` (0.057) and has
no reason left to split on `sleep_hours` (0.002). The column is close to a
restatement of one already ranked above it.

Two of the three strongest signals, class size and sibling count, are outside a
student's control. The model identifies who is at risk, not what to change.

## Limitations

- **Accuracy is measured across all students, but the school only cares about
  the weak ones.** Error on the low-scoring tail has not been evaluated
  separately, and that is the metric that actually matters for this use case.
- **The importance numbers flatter columns with many different values.** Of
  the ways to measure importance, this one counts how often a column gets used
  to split the data. A column with 18 possible values like `class_size` offers
  far more places to split than one with 2 like `age`, so it picks up a higher
  score partly for that reason alone. A fairer check is to shuffle one column
  at random and see how much worse the model gets, which is called permutation
  importance. It was not run here.
- **The deduplication rule is a judgement call.** Where two rows for one
  student both have a recorded `attendance_rate`, the lower one is kept. The
  alternative choices were not tested.
- **The results come from one train/test split.** The three tree models land
  within 0.25 RMSE of each other, which is close enough that another seed could
  reorder them. Repeating across several seeds would settle it.
- **The model shows what goes together, not what causes what.** Small classes
  and high scores appear together in this data, but that is not proof that
  shrinking a class would raise anyone's marks. The model is for spotting which
  students look at risk, not for deciding what the school should change.