# Fitting the world to the model: what it costs, and who it serves

*A one-page summary of the preprint by [Author].*

**The idea.** People and animals don't only adapt to their world. They also change it to suit their habits: paths worn into grass, tools left where they are used, forms that only make sense to the office that designed them. The changes last only as long as someone keeps them up. Earlier work showed that this kind of world-fitting happens. This study asks what it costs, and who pays.

**What we did.** We built a small simulated world and wrote down our predictions before running it.
- Items sit on a ring. Learning agents try actions on them and learn only one thing: whether the world refused the action.
- When an action works, the item shifts a little toward what the agent already expected, so over time the agents reshape their world.
- We then raised new agents inside these reshaped worlds and compared them with agents raised in the untouched world.

**What we found.**
1. **Upkeep.** How much gets built follows a simple rule: the number of builders, times how hard each pushes, divided by how fast things drift back. When the builders stop, it fades. Bigger networks for the builders (within the sizes tested) didn't change it.
2. **Built for one kind of reader.** A reshaped world is much simpler for readers built like the builders than for outsiders: almost all of the simplification disappears for an outsider. Some of the benefit is shared by everyone, but most of the structure is readable only with the builders' "key".
3. **Newcomers don't learn what the world no longer shows.** Agents who already knew the world lost nothing when they moved into a reshaped one. But anyone learning inside it, from scratch or halfway, failed to pick up what reshaping had removed.
4. **A quiet trap.** Agents raised in a reshaped world feel *less* surprised at home than agents raised in the untouched world. Yet beyond the reshaped region they are confidently wrong about two to four times as often. When upkeep stopped and the world snapped back, they earned 42–73% less. The reason: learners normally test themselves just past the edge of what they know, and reshaping removes exactly those test cases.

**What didn't work.** Several predictions failed, and we say so:
- the code-length benefit was mostly shared rather than builder-specific;
- the agreement test was out of reach, because agents agree on a lot even in the untouched world;
- the damage after the snap-back showed up in earnings, but the jump in surprise was smaller than predicted;
- one planned comparison couldn't be run;
- the design can't say anything about arbitrary social conventions, because agents never choose between options.

**Why it might matter.** When the builders of an artifact are much more capable than its readers, the readers can end up as outsiders without the key, or as newcomers who never learn what the artifact hides. Machine-checked mathematics too long for anyone to read is a current case. This study didn't test capacity directly; we describe the experiment that would.

**Open materials.** The predictions, code and every number are public: code under the MIT licence, text under CC BY 4.0.
