<img class="headerimg" src="img/04-header.jpg" alt="Photo of a person opening bottles of sparkling wine with a hammer and nail.">
<div class="chapter">Chapter 4</div>

# When to use Machine Learning

Machine learning is now commonly used in software products of all kinds. However, machine learning is not appropriate for all problems. Sometimes, the adoption of machine learning seems more driven by an interest in the technology or marketing, rather than a real need to solve a specific problem. Adopting machine learning when not needed may be a bad idea, as machine learning brings a lot of complexity, costs, and risks to a system, which can be avoided when simpler options suffice. 



As a scenario, we will discuss personalized music recommendations, such as Spotify automatically curating personalized playlists for each subscriber.

<figure>

![A screenshot of the recommendations made by Spotify showing several personalized mixes with select artists.](./img/04-spotify.png)

<figcaption>

Example of automated personalized music recommendations “Made For You” offered as multiple curated playlists for each subscriber in Spotify.

</figcaption>
</figure>

## Problems that Benefit from Machine Learning

With the hype around machine learning, machine learning can seem like a tempting solution for many problems, but it is not always appropriate. Machine learning involves substantial complexity, cost, and risk, as discussed throughout this book. Among others, teams need considerable expertise to build and deploy models and integrate those models into products. In addition, the learned models are fundamentally unreliable and may make mistakes, so substantial effort is needed to evaluate the product and mitigate risks. 

In general, if it is possible to avoid using machine learning and use hand-coded algorithms instead, it is very often a good idea to do so. However, there are certain classes of problems for which machine learning seems worth the effort. In his book, Geoff Hulten considers the following classes:

**Intrinsically hard problems.**
 For problems we simply do not know how to solve programmatically, machine learning can discover patterns and strategies to solve them regardless. In particular, tasks that mirror human perception tend to be hard, such as natural language understanding, identifying speech in audio and objects in images, and also predicting music preferences. These tasks are complex, and we usually do not fully understand how the original perception works. Pre-machine-learning attempts to program solutions for such tasks have usually made only limited progress. Machine learning may not work for all intrinsically hard problems, but it can be worth a try, and many amazing recent achievements have shown the potential of machine learning.

**Big problems.**
 For some problems, hand-crafting a program to solve a problem might be possible, but it would be so complex and large that manually maintaining it becomes infeasible. Resolving conflicts between multiple hard-coded rules and handling exceptions can be especially tedious. For example, we might attempt to manually encode music recommendations, but there are so many artists and tracks that rule-based heuristics might become unmaintainable. Similarly, manually curating directories of websites has been [tried](https://en.wikipedia.org/wiki/Web_directory) in the early days of the internet, but it simply did not scale, whereas (ML-based) search engines have dominated ever since. For such large problems, automated learning of rules may be easier than writing and maintaining those rules manually.

**Time-changing problems.**
 For problems where the inputs and solutions change frequently, machine learning may be able to more easily keep up if suitable data is available. For example, in music recommendations, individual preferences and music trends change over time, as does what music is available. Change is constant, and even if we had hardcoded recommendation rules, it would be tedious to update them regularly. For such time-changing problems, building a (complex) ML-based solution that can automatically update the system may be easier.

## Tolerating Mistakes and ML Risk

Machine learning should only be used in applications that can tolerate mistakes. In essence, machine-learned models are unreliable functions that often work but sometimes make mistakes. It can even be hard to define what it would mean for them to be correct in the first place, as we will discuss in depth in chapter *[Model Quality](15-model-quality.md)*. 

In some settings, mistakes are simply acceptable. For example, music recommendations are not critical, and subscribers can likely tolerate (some) poor recommendations and benefit from recommendations even if not all are equally good. However, also seemingly harmless predictions can cause harm, for example  music recommendation systematically discriminating against Hispanic subscribers or LGBTQ+ artists.

In addition, sometimes harm from wrong model predictions can be avoided by designing mitigation mechanisms around the model that reduce risk to an acceptable level. For example, artists affected by discriminatory music recommendations may be provided a mechanism to report issues, upon which designers can tweak models or the logic processing the model predictions. Humans can also be involved earlier in the decision process, such as radiologists overseeing the prediction of a cancer prognosis model, and physicians conducting a less invasive and harmful biopsy as a non-ML confirmation before scheduling surgery. Mitigating mistakes requires careful design and evaluation of the system, as we will discuss in chapter *[Planning for Mistakes](07-planning-for-mistakes.md)*. In some cases, mitigations may be costly and undermine the benefits of automation with machine learning; in some cases, we may simply decide that we cannot build the system safely and should not build it at all. In the end, system designers need to carefully weigh the benefits of the system with the costs of its mistakes and the overall risks it poses. 

If the correctness of a component is essential, machine learning is not a good match, especially if there is a specification of correct behavior that we can implement with traditional code. For example, we would not want to use machine-learned components when tabulating information in accounting systems or when transmitting control signals in an airplane, where we can specify the correct expected behavior precisely, and where behaving correctly is crutial.

## Continuous Learning

Using machine learning usually requires access to training and evaluation data for the task. Getting data of sufficient quantity and quality can be a substantial bottleneck and cost driver in a project. In particular, training models for intrinsically hard problems can require substantial amounts of data. While foundation models may seem to alleviate the need for training data somewhat, developers still need some data to design prompts and validate they work effectively for relevant inputs.

Machine learning can be particularly effective in settings, where we have *continuous access to data* and can improve and update the model over time. The previously mentioned *machine learning flywheel* effect suggests that many systems can benefit from observing users over time to collect more data to build better models, which then may attract more users, producing even more data. For example, our music streaming service may monitor in production what music subscribers play and when they skip recommendations—this data observed from operating the system can then improve recommendations.

For time-changing problems, *continuous learning* is essential to keep up with the changing world. Here, we continuously need fresh training data to retrain the model regularly. For example, our music streaming service might deliberately suggest new music to random users to collect data about which kind of users may like it.

Without access to data or no mechanism to observe data continuously in time-changing problems, building a machine-learning-based solution may not be feasible.

## Costs and Benefits

In the end, a decision on whether to use machine learning for a problem comes down to comparing costs and benefits. The machine-learning components need to provide concrete benefits to the system that offset the (often substantial) costs and risks involved in building and operating the system. Costs can come from the initial data acquisition, model building, and deployment—or from paying for an external model API. In addition, a machine-learning component can also create substantial cost during operations—for example, when substantial hardware and energy is needed to serve the model or when humans need to oversee operations and intervene in case of model mistakes. On the other hand, benefits can also be substantial, especially when developing breakthrough capabilities that can dominate market segments or create new market segments. For a music streaming service, recommendations may be just a useful feature; for TikTok*,* recommendations created an entirely new social-media user experience.

Both benefits and costs can be challenging to measure or even estimate. On the benefits side, for example, it can be challenging to quantify how much the music recommendations contribute to attracting (or keeping) subscribers to the streaming service. On the cost side, it is notoriously difficult to estimate development and operating costs before building the system. Also, quantifying potential harm and risk from wrong predictions or systematic bias is challenging. In many cases, start-ups simply bet big and hope their machine-learning innovations will bring huge future payoffs, even if they have huge initial development and operations costs. 

Generally, system designers should always have an open mind and explore whether machine learning is actually needed and cost effective. It may be sufficient to use a simple heuristic instead, which may be less accurate and hence have fewer benefits but also much lower costs. In our music streaming scenario, we might consider simple hard-coded heuristics such as simply recommending the most popular songs on the platform from those artists the subscriber has listened to before. We might also consider a simple semi-manual solution involving a few humans working together with the system, for example, asking a few experts to manually curate twenty playlists and recommend the one that most overlaps with a user’s recently played songs. Finally, we can consider the system without the feature—what would the music streaming system look like without personalized recommendations? If the costs outweigh the benefits, we should be ready to stop the entire project. 

## The Business Case: Machine Learning as Predictions

In this book, we primarily focus on engineering rather than business aspects of building products. However, thinking through the business case can help decide whether and when to use machine learning and consider costs and benefits more broadly. A useful framing comes from the book *[Prediction Machines](https://bookshop.org/books/prediction-machines-the-simple-economics-of-artificial-intelligence/9781633695672)*, which frames the key benefit of machine learning as making *predictions* cheaper, which are used as input for manual or automated *decisions*.

At a high level, machine learning is used to make predictions when there is no plausible algorithm to compute a precise result, such as predicting what music a subscriber likes. The term *“prediction”* implies a best-effort approach to *reduce uncertainty* that uses past data but is not necessarily correct—which fits machine-learning characteristics well. With machine learning, we often improve predictions that otherwise human experts might make, intending to provide *more accurate predictions at a lower cost*. 

Predictions are critical inputs for decision-making. More, faster, and more accurate predictions often help to make better decisions. However, predictions alone are not sufficient, as we still need to interpret the predictions to make decisions—this requires *judgment*. Judgment is fundamentally about trading off the relative benefits and costs of decisions and their outcomes, making decisions about risk. For example, after predicting cancer in a radiology image, making a treatment decision still requires weighing the costs of false positives and false negatives with the benefits of correctly detecting cancer early. Judgment is often left to humans (“human in the loop”), but it is also possible to learn to predict human judgment with enough data, for example, by observing how doctors usually act on cancer predictions. In the case of music recommendations, the subscribers make decisions about whether and when to listen to the music suggested by prediction, though we could also envision a system that automatically decides when and what music to play. Automating judgment makes the step toward *full automation*, where the system itself acts on predictions to maximize some goal.

From a business perspective then, machine learning vastly reduces the cost of predictions and often improves their accuracy, compared to prior approaches such as predictions made by human experts. Higher accuracy and lower cost of predictions may allow us to use cheaper and more predictions for *traditional tasks*, such as replacing knowledgeable record-store employees with automated personalized music recommendations. With lower costs for predictions, we can also consider using predictions for *new applications* at a scale where it was cost-prohibitive to rely on humans, such as curating *personalized* music playlists for all subscribers. Cheap and accurate predictions can enable new business models and enable transformative novel *business strategies*. The book illustrates this with a shop that proactively sends customers products that a model predicts the customers will like—this relies on accurate predictions at scale as predictions need to be accurate enough that the benefits from correct predictions outweigh the costs of paying for return shipping of unwanted items. As these examples illustrate, having access to more, cheaper, and more accurate predictions can be a distinct economic advantage.

Automation is desirable but not necessary to benefit from cheap predictions in a business context. Even when humans are still making decisions, they now benefit from more, more accurate, and faster predictions as inputs. For example, subscribers may have an easier time deciding what music to listen to with our recommendations.

When identifying opportunities for *where* machine learning can provide benefits in an organization, we need to identify what existing or new *tasks* use predictions or could benefit from predictions. For each opportunity, we can then analyze the nature of the predictions, how they contribute to the task, and what benefit we could gain from cheaper, faster, or better predictions. We can then explore to what degree humans previously doing the tasks can be supported or replaced with partial or full automation of predictions and decisions. We would then focus attention where the return on investment is highest, considering costs and benefits as discussed.

## Summary

Using machine learning when it is not needed introduces unnecessary complexity, cost, and risk. However, when problems are intrinsically hard, big, and time-changing, machine learning may provide a solution, as long as the solution can tolerate or mitigate risks from wrong predictions, data is available, and the benefits outweigh the costs. To identify opportunities where machine learning can provide business opportunities, it can also be instructive to think of machine learning as a mechanism to provide cheaper predictions, which in turn can help to make better decisions, whether automated or not.

## Further Readings

  * A book chapter discussing when to use machine learning: 🕮 Hulten, Geoff. *[Building Intelligent Systems: A Guide to Machine Learning Engineering](https://bookshop.org/books/building-intelligent-systems-a-guide-to-machine-learning-engineering/9781484234310)*. Apress, 2018, Chapter 2.

  * An excellent book discussing the business case of machine learning: 🕮 Agrawal, Ajay, Joshua Gans, Avi Goldfarb. *[Prediction Machines: The Simple Economics of Artificial Intelligence](https://bookshop.org/books/prediction-machines-the-simple-economics-of-artificial-intelligence/9781633695672)*. Harvard Business Review Press, 2018.

  * A requirements-modeling approach to identify opportunities for machine learning by analyzing stakeholders, their goals, and their decision needs: 🗎 Nalchigar, Soroosh, Eric Yu, and Karim Keshavjee. “[Modeling Machine Learning Requirements from Three Perspectives: A Case Report from the Healthcare Domain](http://www.cs.utoronto.ca/~soroosh/papers/Modeling%20machine%20learning%20requirements%20from%20three%20perspectives%20a%20case%20report%20from%20the%20healthcare%20domain.pdf).” *Requirements Engineering* 26, no. 2 (2021): 237-254.

  * A frequently shared blog post cautioning (among others) to adopt machine learning only when needed: 📰 Zinkevich, Martin. “[Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml/).” Google Blog (2017).




---
*As all chapters, this text is released under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons BY-NC-ND 4.0</a> license.*
*Last updated on 2024-06-12.*
