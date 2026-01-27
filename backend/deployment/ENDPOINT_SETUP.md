# PRISM-4B Endpoint Setup Guide

## Option 1: HuggingFace Inference Endpoints (Web UI)

Since the API is showing payment issues, create the endpoint via the web interface:

1. **Go to:** https://ui.endpoints.huggingface.co/new

2. **Configure:**
   - **Repository:** `kevruss/PRISM-4B`
   - **Endpoint name:** `prism-4b`
   - **Cloud Provider:** AWS
   - **Region:** us-east-1
   - **Instance Type:** GPU - nvidia-a10g (x1)
   - **Security:** Protected (requires token)

3. **Advanced Settings (Important for PEFT):**
   - **Container Type:** Custom
   - **Image URL:** `ghcr.io/huggingface/text-generation-inference:2.0.0`
   - **Environment Variables:**
     ```
     MODEL_ID=/repository
     ```

4. **Click Create Endpoint**

5. **Wait for deployment** (2-5 minutes)

6. **Test:**
   ```bash
   curl https://YOUR_ENDPOINT_URL/generate \
     -H "Authorization: Bearer $HF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"inputs": "What is happening inside you?", "parameters": {"max_new_tokens": 100}}'
   ```

---

## Option 2: Replicate (Alternative)

Replicate might be simpler for PEFT models:

1. **Create account:** https://replicate.com
2. **Push model:** Use Cog to containerize
3. **Get API endpoint**

---

## Option 3: Modal (Serverless)

1. **Install Modal:**
   ```bash
   pip install modal
   modal setup
   ```

2. **Add HF token as secret:**
   ```bash
   modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN
   ```

3. **Deploy:**
   ```bash
   modal deploy backend/deployment/serve_prism_modal.py
   ```

4. **Get endpoint URL** from Modal dashboard

---

## For LMArena Submission

Once you have an endpoint URL, contact LMArena:

1. **Email:** contact@lmsys.org
2. **Form:** https://lmarena.ai (look for "Add Model" or contact)

**Provide:**
- Model name: `PRISM-4B`
- API endpoint URL
- Brief description: "4B parameter model with enhanced reasoning"
- Authentication details (if protected)

---

## Billing Note

Your HuggingFace account shows:
- `canPay: False`
- `billingMode: prepaid`

You likely need to **add prepaid credits** at:
https://huggingface.co/settings/billing

A10G costs ~$1.30/hour. For testing, $10 of credits should be plenty.
