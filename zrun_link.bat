python -u -m vcsample > result/deepwalk_email.txt ^
    --input mydata/email_con_edge.txt ^
    --graph-format edgelist ^
    --model-v app ^
    --model-c deepwalk ^
    --window-size 4 ^
    --exp-times 5 ^
    --link-prediction
pause
