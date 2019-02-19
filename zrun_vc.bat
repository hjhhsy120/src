python -u -m vcsample > result/deepwalk_email.txt ^
--input mydata/email_con_edge.txt ^
--output result/email_con_app_emd.txt ^
--label-file mydata/email_con_label.txt ^
--graph-format edgelist ^
--model-v deepwalk ^
--model-c deepwalk ^
--window-size 10 ^
--exp-times 5 ^
--classification ^
--clf-ratio 0.5 ^
--epochs 5 ^
--epoch-fac 1000
pause

