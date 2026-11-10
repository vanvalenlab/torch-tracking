Model and Datasets
==================

Pre-trained model weights for all deepcell models are available on
[huggingface](https://huggingface.co/).

To access model weights, you will need to [join the vanvalenlab org][vvhf]
on huggingface.
Navigate to the linked page and click on the "Request to join this org" button
in the upper right, which will require you to log in to your huggingface account.

```{admonition} Pre-trained model license terms
:class: attention

Use of pre-trained model weights is subject to terms of a
[modified Apache 2 license][mesmer-lic] with restrictions on commercial,
non-academic use.

**It is the responsibility of the user to ensure they are in compliance with
the license terms!**

Inquiries regarding commercial licensing should be directed to the
[Caltech Office of Technology Transfer][ottcp].
```

[vvhf]: https://huggingface.co/vanvalenlab
[mesmer-lic]: https://github.com/vanvalenlab/deepcell-auth/blob/main/ASSET_LICENSE
[ottcp]: https://innovation.caltech.edu/about-ottcp/contact-us#send-us-a-message

API Key Usage
-------------

Once you have successfully joined the [vanvalenlab huggingface org][vvhf], you
can [create a huggingface token][hf-token] to access model weights.
Navigate to your huggingface account page and select [Access Tokens][hf-create]
from the sidebar.
Create a new token with **Read** access, and assign it to the name `HF_TOKEN`
in your OS environment:

```bash
export HF_TOKEN="<your-token-here>"
```

[hf-token]: https://huggingface.co/docs/hub/en/security-tokens
[hf-create]: https://huggingface.co/docs/hub/main/en/security-tokens#what-are-user-access-tokens

Models
------

By default, the `CellTracker` instance attempts to download the latest version of
the pre-trained weights upon instantiation.
To see all available model versions, see the [Files and versions page][hf-versions]
on huggingface (note: you must be logged in to see the page).

Specific model versions can be downloaded with the [huggingface CLI][hf-cli].

[hf-versions]: https://huggingface.co/vanvalenlab/torch-tracking/tree/main
[hf-cli]: https://huggingface.co/docs/huggingface_hub/en/guides/cli

Datasets
--------

```{warning}
**There is currently no route for a new user to obtain the
`DynamicNuclearNet-tracking` dataset.**

Model weights have completed the move to huggingface; the datasets have not.
As of 2026-11-04:

- The `vanvalenlab` huggingface organization publishes models but
  [no public datasets][vvhf].
- The `users.deepcell.org` access portal, which previously issued the API keys
  that `deepcell.datasets` used to fetch DynamicNuclearNet, was deprecated in
  10-2026 and **has since been removed**.

This affects anyone who wants to train the model, reproduce the published
accuracy figures, or run the dataset-dependent test layer. Those tasks are
currently possible only for people who already hold a local copy of the data.
Existing copies live under `~/.deepcell/tracking/`.

This gap is documented rather than worked around. If you need the dataset,
contact the Van Valen Lab directly.
```

Note that `DynamicNuclearNet-tracking` (the tracking dataset, used here) is
distinct from `DynamicNuclearNet` (a nuclear segmentation model, which this
package does not use) and from `DynamicNuclearNet-segmentation` (the
segmentation dataset, also unused). See `metrics/PROVENANCE.md` for how these names relate.

Legacy `deepcell-auth`
----------------------

`requirements.txt` still lists `deepcell_auth`, from the era when weights were
fetched through the DeepCell portal. **It is no longer reached by any code
path**: nothing under `torch_tracking/` imports it, and `CellTracker` can be
constructed and used with the module blocked from import entirely (verified
2026-11-04).

It is retained for now because the auth migration is still in flight. It should
be removed once the dataset side of the migration lands and nothing depends on
it.
