# conform

## TODO

- rs rb tree impl

- all label work 2d like neural nets

- LabelMap: np.unique, sorted; fallback if labels are already 0..|Y|,

- predict -> dict to coded 3d matrix

- rework metrics

- rework internal APIs:

  * ncm's

  * vectorizing

- rename stuff 

- tests

## Possible features/changes

- ACP

- IRRCM

- alternative output for RRCM (interval width)

- Venn predict flag for returning different probabilities
  (min, max, mean, median, width (?), 
   prediction set, prediction interval)

- NCSBase -> NCSBaseClassifier

- for MCP: epsilons 2d ((!) METRICS (!))

- experiments suite

- ncs.classifier.score -> vectorized until the end

- score_online with override too and with interval

- decision_tree -> apply ... to predict_proba

- scores -> List to optimized search tree for queries

- CPBase -> rs + par

## Dependencies

- numpy

- scipy

- shapely

- infinity
