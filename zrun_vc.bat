python -u -m vcsample > result/deepwalk_cora.txt ^
--input mydata/cora_con_edge.txt ^
--output result/cora_con_app_emd.txt ^
--label-file mydata/cora_con_label.txt ^
--graph-format edgelist ^
--model-v deepwalk ^
--model-c deepwalk ^
--window-size 10 ^
--exp-times 5 ^
--classification ^
--clf-ratio 0.5 ^
--epochs 1 ^
--epoch-fac 1000
pause

