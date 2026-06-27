# D5 — Executive Summary

**Student:** Herend Amila
**Date:** 28.06.2026
**Course:** Modelling in Advanced Data Analytics — FELU
**Professors:** <span style="color:#0F62FE">**Uroš Godnov**</span> · <span style="color:#0F62FE">**Aleš Gorišek**</span>

---

## Can a Computer Predict Which Songs Become Hits on Spotify?

### The Question

We wanted to know: can we build a computer program that listens to the audio characteristics of a song — things like how danceable it is, how energetic, how loud, what tempo — and correctly predict whether it will become popular on Spotify?

We split this into two practical questions:
1. **Will this song be a hit?** (Yes or No, where a hit means a popularity score of 70 or above out of 100)
2. **How popular exactly will it be?** (a number from 0 to 100)

---

### The Data

We used a publicly available dataset from Kaggle containing **114,000 Spotify tracks** from 114 different music genres. For each song, Spotify provides 21 measurements — none of which involve the artist's name or how the song was marketed. These are purely about the sound itself: tempo (speed), energy (intensity), danceability, how acoustic or electronic it sounds, whether it has lyrics or is purely instrumental, its musical key, and so on.

Spotify also provides a **popularity score** (0–100) for each track, calculated from how many times the song has been played recently. This is what we tried to predict.

---

### What We Found

**Most songs on Spotify are not popular.** Only about 1 in 20 songs (5%) scores 70 or above. The majority of tracks sit between 20 and 40 out of 100. This makes the prediction problem genuinely hard.

We trained four different computer models, from simple to complex, and compared them:

| Approach | Ability to identify a hit | Score prediction accuracy |
|----------|--------------------------|--------------------------|
| Random guessing (baseline) | 0% — misses every hit | Predicts the average every time |
| Simple linear model | Finds about 71% of hits, but many false alarms | Barely better than guessing |
| Decision tree | Similar to linear — overfits | Slightly better |
| **XGBoost (our best model)** | **Finds 76% of hits with far fewer false alarms** | **Predicts within ±16.7 points on average** |

Our best model (XGBoost) correctly identifies 76 out of every 100 genuinely popular songs. When it says a song will be popular, it is right about 31% of the time — which sounds low, but is **six times better than random chance** given how rare hits are.

For the score prediction, the model explains about **38% of what drives popularity**. The other 62% comes from things that have nothing to do with the audio — who the artist is, how big their following is, what label released the song, when it came out, and how well it was promoted.

---

### What We Would Recommend

**For independent artists and producers:**
Focus on the audio features that matter most. Our analysis found that genre, energy level, danceability, and loudness are among the strongest predictors. Songs that are energetic, danceable, and fall within commercial genres (pop, hip-hop, dance) score consistently higher. This does not mean every song should sound the same — but if commercial success is the goal, these patterns are real.

**For record labels and A&R teams:**
Use this kind of model as a first filter, not a final judge. It can quickly screen thousands of demo submissions and flag the ones with audio profiles similar to recent hits — saving time on the initial triage. Human ears and industry judgment should make the final call.

**For streaming platforms:**
Popularity models trained on audio features can support recommendation and playlist curation systems, especially for newly uploaded tracks that have no listening history yet.

---

### What You Should NOT Trust This Model to Do

- **Predict the next viral hit.** Virality depends on timing, social media, and cultural moments — none of which are in the audio data.
- **Evaluate artist originality or artistic quality.** The model rewards commercial conventions, not creativity. A genuinely groundbreaking song that sounds unlike anything before it will score poorly.
- **Replace human music industry expertise.** A model trained on past data cannot anticipate shifts in what audiences will want next year.
- **Work reliably for niche genres.** With only 5% of songs being "popular" overall, the model struggles on genres where popularity is defined differently or where the sample size is small.

---

### Conclusion

Audio features alone can tell you a meaningful amount about a song's commercial potential — but not everything. Our model is a useful tool for narrowing the field, not for making final decisions. The honest finding is that **what makes a song popular is about 38% sound and 62% everything else**.
