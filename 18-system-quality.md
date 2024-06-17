<img class="headerimg" src="img/18-header.jpg" alt="A photo of a half completed wooden puzzle with few large pieces all in the shape of animals. Hands are visible in the foreground and a completed puzzle in the background.">
<div class="chapter">Chapter 18</div>

# System Quality

As discussed throughout this book, machine learning contributes components to a software product. While we can test machine-learned models in isolation, test data quality, test data transformations in machine-learning pipelines, and test the various non-ML components separately, some problems will only arise when integrating the different components as part of a system. 

As explored in chapter *[Setting and Measuring Goals](05-setting-and-measuring-goals.md)*, machine-learning components optimize for goals that ideally support the system goals but do not necessarily align perfectly. It is important to evaluate how the system as a whole achieves its goal, not just how accurate a model is. In addition, since mistakes from machine-learned models are inevitable, we often deliberately design the overall system with safeguards, as discussed in chapter *[Planning for Mistakes](07-planning-for-mistakes.md)*, so anticipated component failures do not result in poor user experiences or even safety hazards. Again, we need to evaluate the system as a whole and how it serves its purpose even in the presence of wrong predictions. In software engineering, end-to-end testing of the entire system is known as *system testing*; testing the system from an end-user’s perspective is also known as *acceptance testing*. 

System and acceptance testing are the last steps in the *V-model*, as discussed in chapter *[Quality Assurance Basics](14-quality-assurance-basics.md)*, bringing the evaluation back to the requirements. It is a good idea to plan what system-wide tests to perform later when first soliciting the requirements is a good idea. This way, requirements can be described as concrete criteria that can later be tested, forcing stakeholders to think about what evidence of system quality they would like to see to accept the system. 

## Limits of Modular Reasoning

Just because components work well in isolation does not guarantee that the system works as expected when those components are integrated. 

**Feature interactions.**
 Often, we made unrealistic assumptions when we decomposed the system, overlooking interactions that do not align easily with module boundaries. In traditional software systems, this is often known as *feature interactions* or *emergent properties*, where the behavior of a composed system may be surprising compared to what we expect from component behavior.  

As a classic example of feature interactions, consider a building safety system that integrates flood control and fire control components from different vendors. Both components were developed and tested individually, and they work as specified. However, when combined in the same building, surprising interactions can happen: when a fire is detected, the fire-control component correctly activates sprinklers, but the flood-control component may then detect the sprinkler water as flooding and shuts down water to the building, undermining the fire-control component. Here, both components interact with the same physical resource in conflicting ways—water. This conflict is not visible from the components’ specifications individually and may only be detected when the components are composed in a specific system.

In traditional software systems, integration and system testing are important to discover unanticipated feature interactions. Better requirements engineering and design can help to some degree to anticipate interactions and plan for them, usually by designing resolution strategies into the overall system, such as giving fire control priority over flood control.

**Change anything changes everything.**
 In machine learning, interactions among components are arguably worse, as we may be composing multiple components that we do not fully understand and where we each expect some wrong answers. Practitioners often speak of the CACE principle: *changing anything changes everything*. A change in training data may affect how the entire system performs, an update in one model may affect data used by another model, and so forth.

Consider the sequential composition of three machine-learned models to generate captions for images introduced in chapter *[Deploying a Model](10-deploying-a-model.md)*. In this approach to image captioning, an object detector identifies objects in the image, a language model suggests many different sentences using those objects, and a ranking model picks the sentence that best fits the picture. Each of the models can be evaluated separately for accuracy using different datasets, but the accuracy of the overall image captioning problem can only be evaluated from the composed system. Each model contributes to the overall solution, but each model has inaccuracies, some of which may be compensated for by other models and some may be exacerbated by the integration with others. Experiments by Nushi et al. even showed that improving the accuracy of one component could make the overall caption worse, since it triggered problems in other components. The quality of the overall solution can only be assessed once the models are integrated.

<figure>

![An illustration of a three step process taking an image and passing it through an object detector, a language model, and a caption reranker to produce a caption. Each step is illustrated with photo of a building and several examples of captions resulting in the caption "Path between buildings sitting on trees" which does not match the original input image.](./img/18-interaction-image-captioning.svg)

<figcaption>

An example of a poor caption generated with an image-captioning system using a three-step sequential architecture of object detector, language model, and caption reranker. Each model makes mistakes, and it is impossible to assign blame to any single component: the object model wrongly identifies a tree in the image albeit with low confidence, the language model creates sentences including ones mentioning trees but also generates sentences with poor common-sense understanding, and the caption reranker picks one of the poorer sentences. Based on observations in 🗎 Nushi, Besmira, Ece Kamar, Eric Horvitz, and Donald Kossmann. “[On Human Intellect and Machine Failures: Troubleshooting Integrative Machine Learning Systems.](https://arxiv.org/abs/1611.08309)” In AAAI Conference on Artificial Intelligence. 2017.

</figcaption>
</figure>

Unpredictable interactions among the various machine-learning components within software systems are another reason to emphasize integration testing and system testing.

**Interactions among machine-learning and non-ML components.**
 As discussed throughout this book, production systems are composed of multiple components interacting with one or more machine-learned models. To support the model, the system usually has several additional model-related components, such as a pipeline to train the model, a storage system for the training data, a subsystem generating training or telemetry data, and a system to monitor the model. In addition, the non-ML parts of the system that use and process or display the prediction are important for implementing safeguards (see chapter *[Planning for Mistakes](07-planning-for-mistakes.md)*) and designing suitable user interfaces.

<figure>

![An architectural diagram of a transcription service showing several non-ML and ML components. Non-ML components include the user interface, user accounts, audio upload, playment, and cloud processing. ML components are data labeling, feature store, training data database, model inference, the ML pipeline, and model monitoring. The ML components are connected through arrows and to non-ML components illustrating the interactions, such as data from the user interface flowing into the training data database as telemetry.](./img/18-architecture-speech-with-components-details.svg)

<figcaption>

A more detailed architecture diagram of the transcription service from chapter *[From Models to Systems](02-from-models-to-systems.md)*, illustrating how even a system with a single model has multiple interacting components related to that model, including data stores, the ML pipeline, and monitoring components.

</figcaption>
</figure>

It is important to test that these components interact with each other as intended. In the transcription-system example, does the telemetry mechanism in the service’s interface correctly write manual edits to the transcript into the database used for training? Does the ML pipeline read the entire dataset? Does the pipeline correctly deploy the updated model? Would the monitoring system actually detect a regression in model quality? And is the user-interface design suitable for communicating the uncertainty of the model’s transcriptions to set reasonable expectations and avoid disappointment or hazards from misrecognized words? 

Testing the integration of components is particularly important when it comes to safety features. In designing safety features, we usually plan for certain kinds of interactions, where one component may overrule another—these intended interactions usually cannot be tested only at the level of individual components, but we need to ensure that the composed system is effective. For example, tests should ensure that pressure sensors in the train’s doors and corresponding logic correctly overwrite predictions from a vision model, to avoid trapping passengers between train doors when the model fails.

## System Testing

The quality of the entire system is typically evaluated in terms of whether it meets its requirements and, more generally, the needs of its users. Many qualities, such as usefulness, usability, safety, and all qualities discussed in part VI, “Responsible ML Engineering,” since they fundamentally rely on how the system interacts within the real world.

**Manual testing.**
 It is often difficult to set up automated tests of a system as a whole. It is common to test a system manually, where a tester interacts with the system as a user would. A tester typically starts the system and interacts with it through its user interface to complete some tasks. In our transcription example, the tester may create an account, upload an audio file, pay, and download the transcript. If the system interacts with the real world through sensors and actuators, system tests are often performed in the field under realistic conditions. For the automated train door example, a tester might install the system in a real train car or a realistic mockup and actually step between the doors and deliberately change body positions, clothing, and light conditions.

Ideally, system tests are guided by system requirements as described by the V-model. When requirements are provided as [use cases](https://en.wikipedia.org/wiki/Use_case) or [user stories](https://en.wikipedia.org/wiki/User_story) (two forms of requirements documentation describing interactions with the system), testers usually follow the interaction sequences outlined in them for their tests, checking both for successful interactions and correct handling of anticipated problems. Coverage can be evaluated in terms of how well the requirements have been evaluated with (manual) tests.

**Automating system tests.**
 System tests can also be automated, but this often requires some work, as we need to automate or simulate user interactions and detect how the system interacts with the environment. Frameworks like [Selenium](https://www.selenium.dev/) can be used to program user interactions with a system—a sequence of clicks on user interface elements, keyboard sequences to fill a form, and assert statements that can check that the user interface displays the expected output. For mobile apps, test services exist that film physical phones to observe how apps display their interfaces throughout interactions on different hardware. For systems interacting with the physical world, we can feed a sequence of recorded sensor inputs into the system, though this does not allow testing interactions through the environment and feedback loops where sensor inputs depend on prior decisions (e.g., whether to close the door of a train). More sophisticated test automation in the physical world can be achieved with automated tests in controlled test environments such as a test track (but this may require manual repair of the environment if the test fails), or system tests may be possible in simulation. While some automation of system tests is possible, it is usually expensive.

**Acceptance testing.**
 Beyond testing specific interactions with the entire system described in the requirements, acceptance testing focuses on evaluating the system broadly from a user’s or business perspective. Acceptance tests intentionally avoid focusing on technical details and implementation correctness, but evaluate the system as a whole for a task. Returning to the distinction between validation and verification discussed in chapter *[Model Quality](15-model-quality.md)*, system testing *verifies* that the implementation of the entire system meets the system specifications (“that we build the system right”), whereas acceptance testing *validates* that the system meets the user’s needs (“that we build the right system”).

Acceptance testing often involves user studies to evaluate the system’s usability or effectiveness for achieving a task. For example, we could conduct an evaluation in a lab, where we invite multiple non-technical users to try the transcription system to observe how quickly they learn how to use it and whether they develop an appropriate mental model of when to not trust the transcripts. Acceptance testing might be performed by the customer directly in a real-world setting.

**Testing in production.**
 To fully evaluate a system in real-world conditions, it might be possible to test it in production. Since the system is deployed and used in production, we evaluate the entire system, how real users interact with it for real tasks, and how it interacts with the real environment. Production use may also reveal many more corner cases that may be difficult to anticipate or reproduce in an offline testing environment. For example, whereas testers of a transcription service may not try audio files with various dialects or may not try uploading very small or very large files to the system, actual users may do all those things, and it is valuable to notice when they fail. Importantly though, all testing in production comes with the risk that failures affect users and may cause poor experiences or even harm in the real world.

Testing in production needs to be carefully planned and designed, including (a) planning what telemetry to collect to evaluate how well the deployed system reaches the overall system goals and (b) protecting users against the consequences of quality problems. We will discuss this in detail in chapter *[Testing and Experimenting in Production](19-testing-and-experimenting-in-production.md)*

## Testing Component Interactions and Safeguards

Beyond testing the entire system end to end, it is also possible to test the composition of individual components and to test subsystems, which is known as integration testing. Integration testing is the middle ground between unit testing of individual components and system testing of the entire system.

Integration tests generally resemble unit tests and are written and automated with the same testing frameworks. The distinguishing difference is that integration tests execute instructions that involve multiple components. An integration test might call multiple functions (ML or non-ML) and pass the result of one function as the input to another function, asserting whether the overall result meets the expectation. For example, in the image-captioning system, we could test the composition of two or all three models—without testing the rest of the system, like the user interface and the monitoring infrastructure. Focusing on infrastructure integration, we could test that the upload of a new model from the ML pipeline to the serving infrastructure works, and we could test that a crowd-sourced data labeling service is correctly integrated with the ML pipeline so that new models are trained based on new labels produced by that service.

In practice, interaction tests are particularly important for safeguards and error handling to recover when one component fails. As discussed earlier, we can test that a train door’s pressure sensors correctly overwrite the vision model. Integration tests here can use inputs that deliberately fail one component, or we can inject erroneous behavior with *stubs* as discussed in chapter *[Pipeline Quality](17-pipeline-quality.md)*. 

<figure>

```js
// making predictions with an ensemble of models
function predict_price(data, models, timeoutms) {
  // send asynchronous REST requests all models
  const requests = models.map(model => rpc(model, data, {timeout: timeoutms}).then(parseResult).catch(e => -1))
  // collect all answers and return average if at least two models succeeded
  return Promise.all(requests).then(predictions => {
    const success = predictions.filter(v => v >= 0)
    if (success.length < 2) throw new Error("Too many models failed")
    return success.reduce((a, b) => a + b, 0) / success.length
  })
}

// integration tests for ensemble of models
const timeout = 500, M1 = "http://localhost:3000/predict", ...
beforeAll(() => {
  // launch model 1 API at address M1
  // launch model 2 API at address M2
  // launch model API with timeout at address M3
}
afterAll(() => { /* shut down all model APIs */ }
test("success despite timeout", async () => {
  const start = performance.now();
  const val = await predict_price(input, [M1, M2, M3], timeout)
  expect(performance.now() - start).toBeLessThan(2 * timeout)
  expect(val).toBeGreaterThan(0)
})
test("fail on too many timeouts", async () => {
  const start = performance.now();
  const val = await predict_price(input, [M1, M3, M3], timeout)
  expect(performance.now() - start).toBeLessThan(2 * timeout)
  expect(val).toThrow()
})
```

<figcaption>

An example of two integration tests for a Javascript implementation of an ensemble model and that it is supposed to return a timely response even if one model fails or responds too slowly. The different components, including a stub injecting a network timeout, are launched as part of the test setup (“beforeAll”).

</figcaption>
</figure>

## Testing Operations (Deployment, Monitoring)

Beyond testing the core functionality, deliberately testing the infrastructure to deploy, operate, and monitor the system can be prudent.

**Deployment.**
 It can be worth the test that automated deployments of the entire system (not just the model) work as expected, especially if regular updates are expected. This can include testing the deployment steps themselves, testing the error handling for various anticipated problems during deployment, and testing whether the monitoring and alerting infrastructure notices (deliberately injected) deployment problems. Infrastructure can be tested for robustness, similar to the machine-learning infrastructure tests discussed in chapter *[Pipeline Quality](17-pipeline-quality.md)*. 

**Robust operations.**
 Production environments create real-world problems that can be difficult to anticipate or simulate when testing offline with stubs, especially for large distributed systems. With ideas like *chaos engineering*, engineers intentionally inject faults *in production systems* to evaluate how robust the system is to faults under real-world conditions. Chaos engineering focuses particularly on faults within the infrastructure, such as network issues and server outages. We will discuss chaos engineering in chapter *[Testing and Experimenting in Production](19-testing-and-experimenting-in-production.md)*.

**Monitoring and alerting.**
 Finally, monitoring and alerting infrastructure is notoriously difficult to test. Incorrect setup of monitoring and alerting infrastructure can let actual problems go undetected for a long time. It is technically possible to set up automated tests that check monitoring and alerting code, such as writing test code to first launch the system and monitor, to then inject a problem into the system, and to finally assert that an alert is raised within five minutes. However, setting up such tests for monitoring code can be tedious and somewhat artificial, given that most monitoring infrastructure evaluates logged behavior over a longer period of time in fairly noisy settings. Most organizations that are serious about evaluating monitoring and alerting use *“fire drills”* or *“smoke tests”* where a problem is intentionally introduced in a production system to observe whether the monitoring system correctly alerts the right people. Such fire drills are usually performed manually, carefully injecting problems to not disrupt the actual operation too much. For example, testers might feed artificial log data that would indicate a problem into the problem. Fire drills need to be scheduled regularly to be effective.

## Summary

Testing cannot end with a unit test at the component level, and evaluating model and data quality in isolation is insufficient for assuring that the entire system works well when used by users in the real world for real tasks. Integration testing, system testing, and acceptance testing are important, even if they may be tedious and not particularly liked by developers. Even with a trend toward testing in production, which tests the entire system under real-world conditions, performing some integration testing and some system testing offline before deployment may catch many problems that arise from the interaction of multiple components. With the lack of specifications for machine-learned models (which requires us to work with models as unreliable functions), integration testing and system testing become even more important.

## Further Readings

  * Testing, including integration testing and system testing, is covered in many books on software testing, such as: 🕮 Copeland, Lee. *[A Practitioner's Guide to Software Test Design](https://bookshop.org/books/a-practitioner-s-guide-to-software-test-design/9781580537919)*. Artech House, 2004. 🕮 Aniche, Mauricio. *[Effective Software Testing: A Developer's Guide](https://bookshop.org/books/effective-software-testing-a-developer-s-guide/9781633439931)*. Simon and Schuster, 2022; and 🕮 Roman, Adam. *[Thinking-Driven Testing](https://bookshop.org/books/thinking-driven-testing-the-most-reasonable-approach-to-quality-control/9783319731940)*. Springer, 2018.

  * An in-depth discussion of the composed image captioning system and the difficulty of assigning blame to any one component, and how local improvements of components do not always translate to improvements of overall system quality: 🗎 Nushi, Besmira, Ece Kamar, Eric Horvitz, and Donald Kossmann. “[On Human Intellect and Machine Failures: Troubleshooting Integrative Machine Learning Systems.](https://arxiv.org/abs/1611.08309)” In *AAAI Conference on Artificial Intelligence,* 2017. 

  * An excellent discussion of different quality criteria in ML-enabled systems that explicitly considers a system perspective and an infrastructure perspective as part of an overall evaluation: 🗎 Siebert, Julien, Lisa Joeckel, Jens Heidrich, Koji Nakamichi, Kyoko Ohashi, Isao Namba, Rieko Yamamoto, and Mikio Aoyama. “[Towards Guidelines for Assessing Qualities of Machine Learning Systems](https://arxiv.org/pdf/2008.11007.pdf).” In *International Conference on the Quality of Information and Communications Technology*, pp. 17–31. Springer, Cham, 2020.

  * An extended discussion of how past approaches to managing feature interactions might design solutions for systems with machine-learning components: 🗎 Apel, Sven, Christian Kästner, and Eunsuk Kang. “[Feature Interactions on Steroids: On the Composition of ML Models](https://arxiv.org/abs/2105.06449).” *IEEE Software* 39, no. 3 (2022): 120–124.

  * A book covering monitoring and alerting strategies in depth. While not specific to machine learning, the techniques apply to observing ML pipelines just as well: 🕮 Ligus, Slawek. *[Effective Monitoring and Alerting](https://bookshop.org/books/effective-monitoring-and-alerting-for-web-operations/9781449333522)*. O'Reilly Media, Inc., 2012.

  * A great book about building and operating systems reliably at scale, including a chapter on how to test infrastructure: 🕮 Beyer, Betsy, Chris Jones, Jennifer Petoff, and Niall Richard Murphy. *[Site Reliability Engineering](https://sre.google/books/)*. O’Reilly, 2017.




---
*As all chapters, this text is released under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons BY-NC-ND 4.0</a> license.*
*Last updated on 2024-06-17.*
