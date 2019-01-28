python -u -m vcsample > result/app_email.txt ^
--input mydata/email_con_edge.txt ^
--output result/email_con_app_emd.txt ^
--label-file mydata/email_con_label.txt ^
--graph-format edgelist ^
--model-v app ^
--exp-times 5 ^
--classification ^
--reconstruction ^
--clustering ^
--modularity
python -u -m vcsample > result/app_email_link.txt ^
--input mydata/email_con_edge.txt ^
--output result/email_con_app_emd.txt ^
--label-file mydata/email_con_label.txt ^
--graph-format edgelist ^
--model-v app ^
--link-prediction
pause
