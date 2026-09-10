# Injection Sentry v3 — canlı deneme

Türkçe prompt-injection dedektörü. Skorlama, model kartındaki ölçümle birebir aynı yoldan
geçer (`infer.py`: `preprocess_v3` + token-id pencereleme + `calib.json` havuzlama ve eşik).

- Model, ölçüm, ön kayıt ve bilinen zayıflıklar: https://huggingface.co/Verm1ion/injection-sentry-v3
- Uygulama: `streamlit_app.py` · bağımlılıklar: `requirements.txt` (CPU torch)

Streamlit Community Cloud'da barındırılır. Lisans cc-by-4.0.
