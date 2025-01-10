# #!/bin/bash
# Each certifair.py run produces two files as side effects: test_predictions.csv and train_predictions.csv
# After each run, preserve these files so they don't get overwritten instantly

mkdir -p predictions

#P2 properties
#base
python certifair.py adult property5 adult_p5_base --lr 0.001 --fr 0.0 --no-bound_loss
mv test_predictions.csv predictions/adult-base-P2-test.csv
mv train_predictions.csv predictions/adult-base-P2-train.csv
python certifair.py german property1 german_p1_base --lr 0.007 --fr 0.0 --no-bound_loss --layers "30,30,1"
mv test_predictions.csv predictions/german-base-P2-test.csv
mv train_predictions.csv predictions/german-base-P2-train.csv
python certifair.py compas property1 compas_p1_base --lr 0.001 --fr 0.0 --no-bound_loss
mv test_predictions.csv predictions/compas-base-P2-test.csv
mv train_predictions.csv predictions/compas-base-P2-train.csv

#global
python certifair.py adult property5 adult_p5_global --lr 0.001 --fr 0.05 --property_loss
mv test_predictions.csv predictions/adult-global-P2-test.csv
mv train_predictions.csv predictions/adult-global-P2-train.csv
python certifair.py german property1 german_p1_global --lr 0.007 --fr 0.01 --property_loss --layers "30,30,1"
mv test_predictions.csv predictions/german-global-P2-test.csv
mv train_predictions.csv predictions/german-global-P2-train.csv
python certifair.py compas property1 compas_p1_global --lr 0.01 --fr 0.1 --property_loss
mv test_predictions.csv predictions/compas-global-P2-test.csv
mv train_predictions.csv predictions/compas-global-P2-train.csv

#local
python certifair.py adult property5 adult_p5_local --lr 0.01 --fr 0.95 
mv test_predictions.csv predictions/adult-local-P2-test.csv
mv train_predictions.csv predictions/adult-local-P2-train.csv
python certifair.py german property1 german_p1_local --lr 0.007 --fr 0.2  --layers "30,30,1"
mv test_predictions.csv predictions/german-local-P2-test.csv
mv train_predictions.csv predictions/german-local-P2-train.csv
python certifair.py compas property1 compas_p1_local --lr 0.01 --fr 0.9 
mv test_predictions.csv predictions/compas-local-P2-test.csv
mv train_predictions.csv predictions/compas-local-P2-train.csv

#P1 properties

#base
python certifair.py adult property6 adult_p6_base --lr 0.001 --fr 0.0 --no-bound_loss
mv test_predictions.csv predictions/adult-base-P1-test.csv
mv train_predictions.csv predictions/adult-base-P1-train.csv
python certifair.py german property2 german_p2_base --lr 0.007 --fr 0.0 --no-bound_loss --layers "30,30,1"
mv test_predictions.csv predictions/german-base-P1-test.csv
mv train_predictions.csv predictions/german-base-P1-train.csv
python certifair.py compas property2 compas_p2_base --lr 0.001 --fr 0.0 --no-bound_loss
mv test_predictions.csv predictions/compas-base-P1-test.csv
mv train_predictions.csv predictions/compas-base-P1-train.csv

#global
python certifair.py adult property6 adult_p6_global --lr 0.001 --fr 0.01 --property_loss
mv test_predictions.csv predictions/adult-global-P1-test.csv
mv train_predictions.csv predictions/adult-global-P1-train.csv
python certifair.py german property2 german_p2_global --lr 0.001 --fr 0.006 --property_loss --layers "30,30,1"
mv test_predictions.csv predictions/german-global-P1-test.csv
mv train_predictions.csv predictions/german-global-P1-train.csv
python certifair.py compas property2 compas_p2_global --lr 0.001 --fr 0.02 --property_loss
mv test_predictions.csv predictions/compas-global-P1-test.csv
mv train_predictions.csv predictions/compas-global-P1-train.csv

#local
python certifair.py adult property6 adult_p6_local --lr 0.01 --fr 0.95 
mv test_predictions.csv predictions/adult-local-P1-test.csv
mv train_predictions.csv predictions/adult-local-P1-train.csv
python certifair.py german property2 german_p2_local --lr 0.007 --fr 0.2  --layers "30,30,1"
mv test_predictions.csv predictions/german-local-P1-test.csv
mv train_predictions.csv predictions/german-local-P1-train.csv
python certifair.py compas property2 compas_p2_local --lr 0.01 --fr 0.9 
mv test_predictions.csv predictions/compas-local-P1-test.csv
mv train_predictions.csv predictions/compas-local-P1-train.csv
