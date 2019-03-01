python -u -m vcsample > result/combine0.8_email_lp.txt ^
--input mydata/email_con_edge.txt ^
--label-file mydata/email_con_label.txt ^
--graph-format edgelist ^
--model-v deepwalk,app ^
--model-c deepwalk,app ^
--app-jump-factor 0.3 ^
--window-size 10 ^
--exp-times 5 ^
--link-prediction ^
--epochs 5 ^
--epoch-fac 1000 ^
--combine 0.8
pause

