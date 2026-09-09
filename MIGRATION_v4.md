# v4 migration checklist

1. Replace the repository contents with this package.
2. Rotate the RapidAPI key if the old one was ever exposed.
3. Add the replacement key as GitHub Actions secret `MATCHSTAT_API_KEY`.
4. Open **Actions → Daily ATP production refresh → Run workflow**.
5. Wait for all steps to finish green, including **Run smoke tests** and **Verify deployable artifacts and data quality**.
6. Reload the Streamlit app. The header should report pipeline/model artifacts from the new refresh and the latest match date should no longer be the stale bootstrap date.

Do not rely on the bundled generated model until the first Matchstat-enabled workflow has completed. The package contains generated artifacts only so the app can boot; the first successful refresh is what produces the current v4 production state.
