const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

function loadQuoteMetaHelpers() {
  const templatePath = path.join(__dirname, '..', 'templates', 'index.html');
  const html = fs.readFileSync(templatePath, 'utf8');
  const start = html.indexOf('function normalizeQuoteSource');
  const end = html.indexOf('function formatTime', start);
  assert.notEqual(start, -1, 'formatQuoteSourceLabel not found');
  assert.notEqual(end, -1, 'formatTime not found after quote helpers');
  const snippet = html.slice(start, end);
  const source = [
    "const translations = {",
    "  quote_source_unknown: 'Quote',",
    "  quote_source_ws: 'WS',",
    "  quote_source_snapshot: 'Snapshot',",
    "  quote_source_summary: 'Summary',",
    "  quote_source_active: 'Active',",
    "  quote_updated_label: 'Updated {time}',",
    "  quote_time_unknown_label: 'Update time unknown',",
    "  quote_time_unknown_title: 'This leg has no verified quote timestamp, so treat it as lower confidence.',",
    "};",
    "const t = (key, params = {}) => {",
    "  let text = translations[key] || key;",
    "  for (const [param, value] of Object.entries(params)) {",
    "    text = text.replace('{' + param + '}', String(value));",
    "  }",
    "  return text;",
    "};",
    "const formatDateTime = (value) => value ? String(value) : '';",
    snippet,
    "return { buildQuoteMeta };",
  ].join('\n');
  return new Function(source)();
}

test('buildQuoteMeta marks quote time as unknown when timestamp is missing', () => {
  const { buildQuoteMeta } = loadQuoteMetaHelpers();
  const meta = buildQuoteMeta({ quote_source: 'snapshot' });

  assert.equal(meta.sourceLabel, 'Snapshot');
  assert.equal(meta.updatedLabel, 'Update time unknown');
  assert.equal(meta.updatedTitle, 'This leg has no verified quote timestamp, so treat it as lower confidence.');
});
