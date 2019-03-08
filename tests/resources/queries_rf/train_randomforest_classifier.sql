select
  train_randomforest_classifier(
    array(age, fare, mhash(embarked), mhash(sex), mhash(pclass))
    , survived
    , '-trees 15 -seed 31 -attrs Q,Q,C,C,C'
  )
from
  ${source}
;
