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

  function nonNegativeNumber(value, fallback) {
    const parsed = finiteNumber(value, fallback);
    return parsed >= 0 ? parsed : fallback;
  }

  function clamp(value, minValue, maxValue) {
    return Math.min(maxValue, Math.max(minValue, value));
  }

  function providerKeySet(customProviderKeys) {
    return new Set(uniqueStrings(customProviderKeys));
  }

  function liveProviderKeys(options) {
    const allProviders = uniqueStrings(options && options.customProviderKeys);
    const supported = uniqueStrings(options && options.liveSupportedProviderKeys);
    if (!supported.length) return allProviders;
    const allProviderSet = new Set(allProviders);
    return supported.filter((key) => !allProviders.length || allProviderSet.has(key));
  }

  function defaultLiveProviderKeys(options) {
    const supported = liveProviderKeys(options);
    const supportedSet = new Set(supported);
    const preferred = uniqueStrings(options && options.defaultLiveProviderKeys)
      .filter((key) => supportedSet.has(key));
    if (preferred.length) return preferred;
    return supported;
  }

  function normalizeScanMode(value) {
    const text = String(value == null ? '' : value).trim().toLowerCase();
    return text === 'live' ? 'live' : 'prematch';
  }

  function selectedBookmakers(options) {
    const useAllBookmakers = options && options.useAllBookmakers === true;
    const providerOnlyMode = options && options.providerOnlyMode === true;
    const scanMode = normalizeScanMode(options && options.scanMode);
    const allBookmakers = uniqueStrings(options && options.allBookmakers);
    const checkedBookmakers = uniqueStrings(options && options.checkedBookmakers);
    const providerSet = providerKeySet(options && options.customProviderKeys);
    const values = useAllBookmakers ? allBookmakers : checkedBookmakers;
    if (scanMode === 'live') {
      if (useAllBookmakers) return defaultLiveProviderKeys(options);
      const supportedSet = new Set(liveProviderKeys(options));
      return values.filter((value) => supportedSet.has(value));
    }
    if (!providerOnlyMode && scanMode !== 'live') return values;
    return values.filter((value) => providerSet.has(value));
  }

  function selectedIncludeProviders(options) {
    const useAllBookmakers = options && options.useAllBookmakers === true;
    const providerOnlyMode = options && options.providerOnlyMode === true;
    const scanMode = normalizeScanMode(options && options.scanMode);
    const bookmakers = uniqueStrings(options && options.bookmakers);
    const providerSet = providerKeySet(options && options.customProviderKeys);
    if (scanMode === 'live' && useAllBookmakers) return defaultLiveProviderKeys(options);
    if (useAllBookmakers) return Array.from(providerSet);
    const eligibleProviderSet = scanMode === 'live'
      ? new Set(liveProviderKeys(options))
      : providerSet;
    const picked = [];
    for (const key of bookmakers) {
      if (!eligibleProviderSet.has(key) || picked.includes(key)) continue;
      picked.push(key);
    }
    if (scanMode === 'live' && !picked.length) return defaultLiveProviderKeys(options);
    if (picked.length || (!providerOnlyMode && scanMode !== 'live')) return picked;
    return Array.from(providerSet);
  }

  function buildServerAutoScanConfigPayload(options) {
    const defaults = options && options.defaults ? options.defaults : {};
    const bookmakers = selectedBookmakers(options);
    const includeProviders = selectedIncludeProviders({
      useAllBookmakers: options && options.useAllBookmakers,
      scanMode: options && options.scanMode,
      providerOnlyMode: options && options.providerOnlyMode,
      customProviderKeys: options && options.customProviderKeys,
      defaultLiveProviderKeys: options && options.defaultLiveProviderKeys,
      liveSupportedProviderKeys: options && options.liveSupportedProviderKeys,
      bookmakers,
    });
    return {
      enabled: true,
      intervalMinutes: nonNegativeNumber(options && options.intervalMinutes, 10),
      payload: {
        scanMode: normalizeScanMode(options && options.scanMode),
        sports: uniqueStrings(options && options.sports),
        allSports: options && options.allSports === true,
        allMarkets: options && options.allMarkets !== undefined
          ? options.allMarkets === true
          : Boolean(defaults.allMarkets),
        stake: positiveNumber(options && options.stake, 100),
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
      scanMode: options && options.scanMode,
      providerOnlyMode: options && options.providerOnlyMode,
      customProviderKeys: options && options.customProviderKeys,
      defaultLiveProviderKeys: options && options.defaultLiveProviderKeys,
      liveSupportedProviderKeys: options && options.liveSupportedProviderKeys,
      bookmakers,
    });
    return {
      apiKey: stringValue(options && options.apiKey, ''),
      scanMode: normalizeScanMode(options && options.scanMode),
      sports: uniqueStrings(options && options.sports),
      bookmakers,
      includeProviders,
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
