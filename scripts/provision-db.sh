#!/usr/bin/env bash
# Create the localllm database and role inside an EXISTING Postgres container,
# for when you already run Postgres and do not want the `db` service from
# docker-compose.yml.
#
# Non-destructive and idempotent: it only ever creates objects named
# `localllm*`, and never touches other databases, roles or data.
#
# Usage:  DB_PASSWORD='…' ./scripts/provision-db.sh
set -euo pipefail

PG_CONTAINER="${PG_CONTAINER:-postgres}"
PG_SUPERUSER="${PG_SUPERUSER:-postgres}"
DB_NAME="${DB_NAME:-localllm}"
DB_USER="${DB_USER:-localllm_user}"
DB_PASSWORD="${DB_PASSWORD:?DB_PASSWORD must be set}"

if ! docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
  echo "error: container '$PG_CONTAINER' is not running" >&2
  exit 1
fi

psql_super() {
  docker exec -i "$PG_CONTAINER" psql -v ON_ERROR_STOP=1 -U "$PG_SUPERUSER" -d postgres "$@"
}

echo "==> ensuring role '$DB_USER'"
psql_super <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$DB_USER') THEN
    CREATE ROLE $DB_USER LOGIN PASSWORD '$DB_PASSWORD';
    RAISE NOTICE 'created role $DB_USER';
  ELSE
    ALTER ROLE $DB_USER LOGIN PASSWORD '$DB_PASSWORD';
    RAISE NOTICE 'role $DB_USER already existed; password rotated';
  END IF;
END
\$\$;
SQL

echo "==> ensuring database '$DB_NAME'"
if psql_super -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1; then
  echo "    database already exists — left as-is"
else
  psql_super -c "CREATE DATABASE $DB_NAME OWNER $DB_USER"
  echo "    created"
fi

echo "==> granting schema privileges"
docker exec -i "$PG_CONTAINER" psql -v ON_ERROR_STOP=1 -U "$PG_SUPERUSER" -d "$DB_NAME" <<SQL
GRANT ALL ON SCHEMA public TO $DB_USER;
ALTER SCHEMA public OWNER TO $DB_USER;
-- pgcrypto provides gen_random_uuid(); creating an extension needs superuser.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
SQL

echo
echo "Done. Databases now present:"
psql_super -tAc "SELECT datname FROM pg_database ORDER BY 1" | sed 's/^/  - /'
