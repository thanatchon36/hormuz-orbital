On 28 February 2026, tanker traffic through the Strait of Hormuz all but stopped.

Daily tanker transits fell from 47.6 to 2.4, a 95% drop. Brent went from $71 to $138 a barrel in six weeks.

That raises a question I couldn't stop thinking about: when missiles hit the Gulf, do oil prices jump because barrels physically stop moving, or because markets start pricing in risk?

The two usually arrive together, so prices alone can't separate them. You need to see the ships.

So I wrote a preprint on it: "Orbital Evidence on Chokepoint Risk."

🛰️ A satellite transit index for Hormuz that fuses Sentinel-1 radar with AIS, so it still counts ships that switch off their transponders
🎯 A spatial strike-exposure index that tells a hit on a refinery apart from a hit on a remote air base
📈 An event-study and local-projection design that splits a price move into lost barrels and risk premium

Early evidence from six episodes, 2019–2026:
• In three of them, Brent jumped 7–11 log points on day 0, then gave most of it back
• Only the 2026 episode, the one where physical flow collapsed, kept climbing: +52.8 by day 20 (placebo p = 0.015)
• In calm weeks, small swings in tanker traffic barely move prices

Six events are enough to motivate hypotheses, not to test them. The radar pipeline is built and tested on simulated scenes; processing real Sentinel-1 passes comes next.

Everything is open:
📖 Paper: https://thanatchon36.github.io/hormuz-orbital/paper/
💻 Code: https://github.com/thanatchon36/hormuz-orbital
🤗 Data: https://huggingface.co/datasets/thanatchon36/hormuz-orbital
🔖 DOI: 10.5281/zenodo.24184102

If you work with SAR, AIS, or energy markets, I'd really value your feedback, especially on detecting vessels in rough sea clutter.

#RemoteSensing #Sentinel1 #EnergyMarkets #OilPrices #Geopolitics #Econometrics #OpenScience
