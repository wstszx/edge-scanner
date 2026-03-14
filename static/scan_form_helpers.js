(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
  root.EdgeScannerScanFormHelpers = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function uniqueStrings(values) {
    const list = Array.isArray(values) ? values : [];
    const seen = new Set();
    const out = [];
    for (const value of list) {
      const text = String(value || '').trim();
      if (!text || seen.has(text)) continue;
      seen.add(text);
      out.push(text);
    }
    return out;
  }

  function stringValue(value, fallback) {
    const text = String(value == null ? '' : value).trim();
    return text || String(fallback == null ? '' : fallback).trim();
  }

  function finiteNumber(value, fallback) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function positiveNumber(value, fallback) {
    const parsed = finiteNumber(value, fallback);
    return parsed > 0 ? parsed : fallback;
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(maxValue, Math.max(minValue, value));
  }

  function providerKeySet(customProviderKeys) {
    return new Set(uniqueStrings(customProviderKeys));
  }

  function selectedBookmakers(options) {
    const useAllBookmakers = options && options.useAllBookmakers === true;
    const providerOnlyMode = options && options.providerOnlyMode === true;
    const allBookmakers = uniqueStrings(options && options.allBookmakers);
    const checkedBookmakers = uniqueStrings(options && options.checkedBookmakers);
    const providerSet = providerKeySet(options && options.customProviderKeys);
    const values = useAllBookmakers ? allBookmakers : checkedBookmakers;
    if (!providerOnlyMode) return values;
    return values.filter((value) => providerSet.has(value));
  }

  function selectedIncludeProviders(options) {
    const useAllBookmakers = options && options.useAllBookmakers === true;
    const providerOnlyMode = options && options.providerOnlyMode === true;
    const bookmakers = uniqueStrings(options && options.bookmakers);
    const providerSet = providerKeySet(options && options.customProviderKeys);
    if (useAllBookmakers) return Array.from(providerSet);
    const picked = [];
    for (const key of bookmakers) {
      if (!providerSet.has(key) || picked.includes(key)) continue;
      picked.push(key);
    }
    if (picked.length || !providerOnlyMode) return picked;
    return Array.from(providerSet);
  }

  function buildServerAutoScanConfigPayload(options) {
    const defaults = options && options.defaults ? options.defaults : {};
    const bookmakers = selectedBookmakers(options);
    const includeProviders = selectedIncludeProviders({
      useAllBookmakers: options && options.useAllBookmakers,
      providerOnlyMode: options && options.providerOnlyMode,
      customProviderKeys: options && options.customProviderKeys,
      bookmakers,
    });
    return {
      enabled: true,
      intervalMinutes: positiveNumber(options && options.intervalMinutes, 10),
      payload: {
        sports: uniqueStrings(options && options.sports),
        allSports: options && options.allSports === true,
        allMarkets: options && options.allMarkets !== undefined
          ? options.allMarkets === true
          : Boolean(defaults.allMarkets),
        stake: positiveNumber(options && options.stake, 100),
        regions: uniqueStrings(options && options.regions),
        bookmakers,
        includeProviders,
        commission: finiteNumber(options && options.commission, defaults.commission || 0),
        sharpBook: stringValue(options && options.sharpBook, defaults.sharpBook || ''),
        minEdgePercent: positiveNumber(options && options.minEdgePercent, defaults.minEdgePercent || 1),
        bankroll: positiveNumber(options && options.bankroll, defaults.bankroll || 1000),
        kellyFraction: clamp(
          positiveNumber(options && options.kellyFraction, defaults.kellyFraction || 0.25),
          0,
          1
        ),
      },
    };
  }

  function buildRunScanPayload(options) {
    const defaults = options && options.defaults ? options.defaults : {};
    const bookmakers = selectedBookmakers(options);
    const includeProviders = selectedIncludeProviders({
      useAllBookmakers: options && options.useAllBookmakers,
      providerOnlyMode: options && options.providerOnlyMode,
      customProviderKeys: options && options.customProviderKeys,
      bookmakers,
    });
    return {
      apiKey: stringValue(options && options.apiKey, ''),
      sports: uniqueStrings(options && options.sports),
      bookmakers,
      includeProviders,
      regions: uniqueStrings(options && options.regions),
      commission: finiteNumber(options && options.commission, defaults.commission || 0),
      allSports: options && options.allSports === true,
      allBookmakers: options && options.useAllBookmakers === true,
      allMarkets: options && options.allMarkets !== undefined
        ? options.allMarkets === true
        : Boolean(defaults.allMarkets),
      stake: positiveNumber(options && options.stake, 100),
      sharpBook: stringValue(options && options.sharpBook, defaults.sharpBook || ''),
      minEdgePercent: positiveNumber(options && options.minEdgePercent, defaults.minEdgePercent || 1),
      bankroll: positiveNumber(options && options.bankroll, defaults.bankroll || 1000),
      kellyFraction: clamp(
        positiveNumber(options && options.kellyFraction, defaults.kellyFraction || 0.25),
        0,
        1
      ),
    };
  }

  return {
    selectedBookmakers,
    selectedIncludeProviders,
    buildServerAutoScanConfigPayload,
    buildRunScanPayload,
  };
});
