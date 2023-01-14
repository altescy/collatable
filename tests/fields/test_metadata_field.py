from collatable.fields.metadata_field import MetadataField


def test_metadata_field() -> None:
    fields = [MetadataField({"id": 123}), MetadataField({"id": 456})]
    output = fields[0].collate(fields)
    assert isinstance(output, list)
    assert len(output) == 2
    assert output[0]["id"] == 123
    assert output[1]["id"] == 456
