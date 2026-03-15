const test = require('node:test');
const assert = require('node:assert/strict');

const helpers = require('../static/scan_form_helpers.js');

test('selectedBookmakers filters non-provider books in provider-only mode', () => {
  const bookmakers = helpers.selectedBookmakers({
    useAllBookmakers: false,
    checkedBookmakers: ['draftkings', 'sx_bet', 'betdex'],
    allBookmakers: ['draftkings', 'sx_bet', 'betdex'],
    providerOnlyMode: true,
    customProviderKeys: ['sx_bet', 'betdex', 'polymarket'],
  });

  assert.deepEqual(bookmakers, ['sx_bet', 'betdex']);
});

test('selectedIncludeProviders returns all providers when all bookmakers is enabled', () => {
  const providers = helpers.selectedIncludeProviders({
    useAllBookmakers: true,
    providerOnlyMode: false,
    bookmakers: ['sx_bet'],
    customProviderKeys: ['sx_bet', 'betdex', 'polymarket'],
  });

  assert.deepEqual(providers, ['sx_bet', 'betdex', 'polymarket']);
});

test('selectedIncludeProviders falls back to all providers in provider-only mode when nothing is picked', () => {
  const providers = helpers.selectedIncludeProviders({
    useAllBookmakers: false,
    providerOnlyMode: true,
    bookmakers: [],
    customProviderKeys: ['sx_bet', 'betdex', 'polymarket'],
  });

  assert.deepEqual(providers, ['sx_bet', 'betdex', 'polymarket']);
});

test('selectedIncludeProviders falls back to all providers in live mode when nothing is picked', () => {
  const providers = helpers.selectedIncludeProviders({
    useAllBookmakers: false,
    scanMode: 'live',
    providerOnlyMode: false,
    bookmakers: [],
    customProviderKeys: ['sx_bet', 'betdex', 'polymarket'],
    defaultLiveProviderKeys: ['sx_bet', 'betdex'],
    liveSupportedProviderKeys: ['sx_bet', 'betdex', 'polymarket'],
  });

  assert.deepEqual(providers, ['sx_bet', 'betdex']);
});

test('selectedBookmakers uses curated live defaults when all bookmakers is enabled', () => {
  const bookmakers = helpers.selectedBookmakers({
    useAllBookmakers: true,
    scanMode: 'live',
    allBookmakers: ['draftkings', 'sx_bet', 'betdex', 'polymarket', 'bookmaker_xyz'],
    checkedBookmakers: [],
    providerOnlyMode: false,
    customProviderKeys: ['sx_bet', 'betdex', 'polymarket', 'bookmaker_xyz'],
    defaultLiveProviderKeys: ['sx_bet', 'betdex', 'polymarket'],
    liveSupportedProviderKeys: ['sx_bet', 'betdex', 'polymarket', 'bookmaker_xyz'],
  });

  assert.deepEqual(bookmakers, ['sx_bet', 'betdex', 'polymarket']);
});

test('buildServerAutoScanConfigPayload derives provider-only config from form state', () => {
  const payload = helpers.buildServerAutoScanConfigPayload({
    sports: ['icehockey_nhl'],
    allSports: false,
    allMarkets: true,
    stake: '75',
    regions: ['us'],
    useAllBookmakers: false,
    checkedBookmakers: ['draftkings', 'sx_bet'],
    allBookmakers: ['draftkings', 'sx_bet', 'betdex'],
    providerOnlyMode: true,
    customProviderKeys: ['sx_bet', 'betdex'],
    commission: '3',
    sharpBook: 'pinnacle',
    minEdgePercent: '1.5',
    bankroll: '2500',
    kellyFraction: '0.4',
    intervalMinutes: '12',
    defaults: {
      allMarkets: false,
      sharpBook: 'pinnacle',
      minEdgePercent: 1,
      bankroll: 1000,
      kellyFraction: 0.25,
      commission: 0,
    },
  });

  assert.deepEqual(payload, {
    enabled: true,
    intervalMinutes: 12,
    payload: {
      scanMode: 'prematch',
      sports: ['icehockey_nhl'],
      allSports: false,
      allMarkets: true,
      stake: 75,
      regions: ['us'],
      bookmakers: ['sx_bet'],
      includeProviders: ['sx_bet'],
      commission: 3,
      sharpBook: 'pinnacle',
      minEdgePercent: 1.5,
      bankroll: 2500,
      kellyFraction: 0.4,
    },
  });
});

test('buildRunScanPayload preserves mixed books outside provider-only mode', () => {
  const payload = helpers.buildRunScanPayload({
    apiKey: 'abc',
    scanMode: 'live',
    sports: ['basketball_nba'],
    regions: ['us', 'eu'],
    useAllBookmakers: false,
    checkedBookmakers: ['draftkings', 'sx_bet'],
    allBookmakers: ['draftkings', 'sx_bet', 'betdex'],
    providerOnlyMode: false,
    customProviderKeys: ['sx_bet', 'betdex'],
    defaultLiveProviderKeys: ['sx_bet', 'betdex'],
    liveSupportedProviderKeys: ['sx_bet', 'betdex'],
    commission: '2.5',
    allSports: false,
    allMarkets: false,
    stake: '100',
    sharpBook: 'pinnacle',
    minEdgePercent: '1',
    bankroll: '1000',
    kellyFraction: '0.25',
    defaults: {
      allMarkets: false,
      sharpBook: 'pinnacle',
      minEdgePercent: 1,
      bankroll: 1000,
      kellyFraction: 0.25,
      commission: 0,
    },
  });

  assert.deepEqual(payload, {
    apiKey: 'abc',
    scanMode: 'live',
    sports: ['basketball_nba'],
    bookmakers: ['sx_bet'],
    includeProviders: ['sx_bet'],
    regions: ['us', 'eu'],
    commission: 2.5,
    allSports: false,
    allBookmakers: false,
    allMarkets: false,
    stake: 100,
    sharpBook: 'pinnacle',
    minEdgePercent: 1,
    bankroll: 1000,
    kellyFraction: 0.25,
  });
});

test('buildRunScanPayload uses curated live defaults when nothing is selected', () => {
  const payload = helpers.buildRunScanPayload({
    apiKey: '',
    scanMode: 'live',
    sports: ['basketball_nba'],
    regions: ['us'],
    useAllBookmakers: false,
    checkedBookmakers: [],
    allBookmakers: ['draftkings', 'sx_bet', 'betdex', 'polymarket', 'bookmaker_xyz'],
    providerOnlyMode: false,
    customProviderKeys: ['sx_bet', 'betdex', 'polymarket', 'bookmaker_xyz'],
    defaultLiveProviderKeys: ['sx_bet', 'betdex', 'polymarket'],
    liveSupportedProviderKeys: ['sx_bet', 'betdex', 'polymarket', 'bookmaker_xyz'],
    commission: '1',
    allSports: false,
    allMarkets: false,
    stake: '100',
    sharpBook: 'pinnacle',
    minEdgePercent: '1',
    bankroll: '1000',
    kellyFraction: '0.25',
    defaults: {
      allMarkets: false,
      sharpBook: 'pinnacle',
      minEdgePercent: 1,
      bankroll: 1000,
      kellyFraction: 0.25,
      commission: 0,
    },
  });

  assert.deepEqual(payload.includeProviders, ['sx_bet', 'betdex', 'polymarket']);
  assert.deepEqual(payload.bookmakers, []);
});
