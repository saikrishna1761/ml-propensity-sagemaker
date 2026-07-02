# Manual AWS Resources

These resources are not managed by Terraform because they are difficult or impossible to import cleanly.

## SageMaker Feature Store — Feature Group

- **Name:** `propensity-features-v1`
- **Region:** `ap-south-1`
- **Reason not in Terraform:** SageMaker Feature Store feature groups have complex lifecycle management and schema definitions that do not map cleanly to Terraform resources. Deleting and recreating would wipe offline store data.
- **How it was created:** `python features/feature_store.py` (runs `create_feature_group()` once)
- **To recreate:** Run `python features/feature_store.py` in a new environment.

## SageMaker Model Package Group

- **Name:** `propensity-model`
- **Region:** `ap-south-1`
- **Reason not in Terraform:** Model package groups accumulate versioned model registrations that cannot be imported into Terraform state without risk of data loss.
- **How it was created:** `python training/register_model.py`
- **To recreate:** Run `python training/register_model.py` after training.
