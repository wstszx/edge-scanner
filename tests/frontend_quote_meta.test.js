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
    "  execution_quality_high: 'Execution: high',",
    "  execution_quality_medium: 'Execution: medium',",
    "  execution_quality_low: 'Execution: low',",
    "  execution_quality_unknown: 'Execution: unknown',",
    "  execution_quality_title: 'Execution confidence: {status}. Flags: {flags}',",
    "  execution_quality_no_flags: 'no execution warnings',",
    "  execution_flag_missing_quote_time: 'missing quote time',",
    "  execution_flag_missing_liquidity: 'missing liquidity',",
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
    "return { buildQuoteMeta, buildExecutionQualityMeta };",
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

test('buildExecutionQualityMeta labels low-confidence execution flags', () => {
  const { buildExecutionQualityMeta } = loadQuoteMetaHelpers();
  const meta = buildExecutionQualityMeta({
    execution_quality: {
      status: 'low',
      flags: ['missing_quote_time', 'missing_liquidity', 'custom_warning'],
    },
  });

  assert.equal(meta.status, 'low');
  assert.equal(meta.label, 'Execution: low');
  assert.deepEqual(meta.flags, ['missing quote time', 'missing liquidity', 'custom warning']);
  assert.equal(meta.title, 'Execution confidence: low. Flags: missing quote time, missing liquidity, custom warning');
});
