env | grep -iE '^(https?_proxy|no_proxy|all_proxy|hf_)' | sed 's/\(HF_TOKEN=hf_.\{4\}\).*/\1.../'
curl -sS -o /dev/null -w "curl -> HF %{http_code}\n" --max-time 10 https://huggingface.co/api/models/google/paligemma-3b-pt-224
python -c "from huggingface_hub import HfApi; print('python -> HF ok:', HfApi().model_info('google/paligemma-3b-pt-224').id)"
