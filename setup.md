# Setup Guide – Daily News Blog

This project uses Google Cloud, Gemini API, and GNews API to automate blog creation and posting.

---

## Required Information

* **Blogger Email**: `factopediablogs@gmail.com`
* **Blogger ID**: `8169847264446388236`
* **Google Cloud Console**: [https://console.cloud.google.com/](https://console.cloud.google.com/)
* **Project Name**: `dailynews`

---

## API Keys

Make sure you have the following API keys:

* `GEMINI_API_KEY` (associated with `factopediablogs@gmail.com`)
* `GNEWS_API_KEY` (associated with `factopediablogs@gmail.com`)

---

## .env Example

Create a `.env` file and add:

```env
GEMINI_API_KEY=your_gemini_api_key
GNEWS_API_KEY=your_gnews_api_key
BLOGGER_ID=8169847264446388236
BLOGGER_EMAIL=factopediablogs@gmail.com
```

