python -u -m nesampler --method node2vec --p 4.0 --q 4.0 --label-file mydata/email_label.txt --input mydata/email_edge.txt --graph-format edgelist --output myresult/email_node2vec2_emd.txt
python pic.py 50 email node2vec_4_4
pause
