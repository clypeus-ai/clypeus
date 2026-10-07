-- Headers every provider request from a scope must carry.
--
-- Stored as a JSON array of `{"name": ..., "value": ...}` objects in a text
-- column, the way `extensions` is stored, because the shape belongs to the
-- host and both SQL backends share one schema. The default is an empty array,
-- not SQL NULL: every scope written before this migration required no headers,
-- which is exactly what the empty list says, and a null would answer "no
-- headers" and "never recorded" differently on every read.
--
-- The value is stored in the clear. It is configuration the scope's owner
-- typed — a session header, a tenant header — and the runner has to send it
-- verbatim; it is not a credential the platform holds only to replay. The
-- in-memory record still redacts it on `Debug`, so a log line that prints the
-- settings cannot print the value.
ALTER TABLE clypeus_settings
    ADD COLUMN headers TEXT NOT NULL DEFAULT '[]';
