-- The shape the turn asked the model's answer to take.
--
-- Stored with the turn rather than kept in memory, because a turn that parks for an
-- approval is resumed by a later request that never saw the original one — and a resumed
-- turn still has to end in the document it was asked for.
--
-- `NOT NULL DEFAULT` rather than a nullable column: every turn written before this one
-- asked for free text, which is what the default says, and a null here would put the
-- "was it text or was it never recorded" question into every read.
ALTER TABLE clypeus_messages
    ADD COLUMN output_format TEXT NOT NULL DEFAULT '{"type":"text"}';
