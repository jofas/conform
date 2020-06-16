# conform

ArXiv preprint: https://arxiv.org/abs/1907.02015

## TODO

- rework internal APIs:

  * ncm's

  * vectorizing

- ncs -> ncm

- rs rb tree impl

- rework metrics

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
