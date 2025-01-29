# Remove the files from Git history
git filter-branch --force --index-filter \
  'git rm -r --cached --ignore-unmatch \
  run_logs/uspt_rnn-b8e1.log \
  run_logs/run14_gith_nmt_transformer_model2_errors.log \
  run_logs/uspt_convs2s-try3.log \
  run_logs/uspt_rnn.log \
  run_logs/uspt_convs2s.log \
  run_logs/uspt_transformer.log \
  data/preprocessed/dblp/dblp.v12.json/splits.json \
  run_logs/uspt_transformer-b8e1.log \
  run_logs/run19_gith_nmt_transformer-gith1.log \
  run_logs/uspt_convs2s-final.log' --prune-empty -- --all

# Clean up the repository
git gc --aggressive --prune=now