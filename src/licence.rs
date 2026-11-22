//! Reading a model's `custom.license` string.
//!
//! The result is deliberately three-valued and **biased to refusal**: only a name that is clearly a well-known permissive licence is
//! [`NameClass::Permissive`]; a name that clearly restricts commercial use is [`NameClass::NonCommercial`]; everything else is
//! [`NameClass::Other`], which the registry treats as "not known to permit commercial use" (refused unless the catalog author sets
//! `commercial_use: "true"`).
//!
//! What is **not** classified as permissive, on purpose: the GPL and LGPL families and other copyleft licences, the OpenRAIL and
//! RAIL variants (`openrail++`, `CreativeML OpenRAIL-M`: they carry use restrictions), MPL and `CC-BY-SA`, and every custom or
//! community licence. They permit commercial use under conditions this crate cannot judge, so a catalog author vouches for them with
//! `commercial_use: "true"`.

/// What the licence name alone says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameClass {
    /// Every alternative named is a recognised permissive licence.
    Permissive,
    /// The name restricts use to non-commercial, research, academic, evaluation or personal purposes.
    NonCommercial,
    /// Anything else, including an empty name.
    Other,
}

/// Lowercase alphanumeric words, with `/ ; |` read as "or" so `MIT/Apache-2.0` is two alternatives (a comma is just punctuation: `Apache License, Version 2.0`), and `v2` read as `2`.
fn words(licence: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (n, part) in licence.split(['/', ';', '|']).enumerate() {
        if n > 0 {
            out.push("or".to_string());
        }
        out.extend(
            part.to_ascii_lowercase()
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(|w| match w.strip_prefix('v') {
                    Some(d) if !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) => {
                        d.to_string()
                    }
                    _ => w.to_string(),
                }),
        );
    }
    out
}

/// Words that, near a word about commerce or profit, say the use is not allowed. English, German, French, Spanish, Italian.
const RESTRICTIVE: &[&str] = &[
    "non",
    "no",
    "not",
    "never",
    "without",
    "prohibited",
    "forbidden",
    "banned",
    "restricted",
    "disallowed",
    "denied",
    "excluded",
    "nicht",
    "kein",
    "keine",
    "keinen",
    "keiner",
    "ohne",
    "sans",
    "pas",
    "sin",
    "senza",
];

/// How many words either side of "commercial" or "profit" a [`RESTRICTIVE`] word counts: "not for commercial use" (2), "commercial use
/// is not permitted" (3). Wide on purpose: a false hit only refuses a model.
const WINDOW: usize = 3;

fn is_commerce_word(w: &str) -> bool {
    w.starts_with("commerc") // commercial, commercially, commerciale
        || w.starts_with("kommerz") // kommerziell
        || w.starts_with("comerc") // comercial
        || w == "profit"
        || w == "profits"
}

/// Words that limit a licence to a kind of use other than commercial: research, academic, evaluation, personal, educational.
fn is_scope_word(w: &str) -> bool {
    matches!(
        w,
        "research" | "academic" | "evaluation" | "personal" | "educational" | "hobby"
    )
}

/// A scope word followed by one of these describes the *release*, not the terms: "research preview", "personal fork".
fn is_release_word(w: &str) -> bool {
    matches!(
        w,
        "preview"
            | "fork"
            | "build"
            | "release"
            | "snapshot"
            | "checkpoint"
            | "demo"
            | "prototype"
            | "variant"
            | "port"
            | "copy"
            | "derivative"
            | "model"
            | "weights"
            | "version"
            | "edition"
    )
}

/// The name says non-commercial by negating commerce or profit: `non-commercial`, `not for commercial use`, `commercial use prohibited`,
/// `non-profit`, the Creative Commons `NC` family, and the same in a few other languages.
fn restricts_commercial_use(w: &[String]) -> bool {
    let has = |x: &str| w.iter().any(|y| y == x);
    if has("noncommercial") || has("nonprofit") || has("nc") {
        // `nc` is the Creative Commons NC family (cc-by-nc-4.0, cc-by-nc-sa, cc-by-nc-nd), matched as a whole word
        return true;
    }
    for (i, word) in w.iter().enumerate() {
        if !is_commerce_word(word) {
            continue;
        }
        let lo = i.saturating_sub(WINDOW);
        let hi = (i + WINDOW + 1).min(w.len());
        if w[lo..hi]
            .iter()
            .enumerate()
            .any(|(k, x)| lo + k != i && RESTRICTIVE.contains(&x.as_str()))
        {
            return true;
        }
    }
    false
}

/// Scope words that are not just describing a release. Returns `(restricting, any)`.
fn scope_words(w: &[String]) -> (bool, bool) {
    let mut restricting = false;
    let mut any = false;
    for (i, word) in w.iter().enumerate() {
        if !is_scope_word(word) {
            continue;
        }
        any = true;
        if !w.get(i + 1).is_some_and(|n| is_release_word(n)) {
            restricting = true;
        }
    }
    (restricting, any)
}

/// Words dropped before matching a licence id: `MIT License`, `Apache License, Version 2.0`, `The Unlicense`, `Apache Software License`.
fn is_filler(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "license"
            | "licence"
            | "licenses"
            | "licensed"
            | "version"
            | "software"
            | "international"
            | "unported"
            | "generic"
            | "universal"
    )
}

/// Ids of licences that permit commercial use without further conditions this crate would need to judge, as hyphen-joined words.
const PERMISSIVE: &[&str] = &[
    "mit",
    "apache",
    "apache-2",
    "apache-2-0",
    "apache2",
    "apache2-0",
    "bsd",
    "bsd-2",
    "bsd-3",
    "bsd-2-clause",
    "bsd-3-clause",
    "new-bsd",
    "modified-bsd",
    "simplified-bsd",
    "revised-bsd",
    "isc",
    "cc0",
    "cc0-1-0",
    "unlicense",
    "zlib",
    "cc-by",
    "cc-by-3-0",
    "cc-by-4-0",
];

/// Whether one alternative (words between `or` / `and`) is a recognised permissive id, optionally followed by words that only describe
/// the release ("research preview", "personal fork").
fn is_permissive_alternative(alt: &[String]) -> bool {
    // "Creative Commons Attribution 4.0" is "cc by 4.0"
    let mut a: Vec<String> = Vec::new();
    let mut i = 0;
    while i < alt.len() {
        if alt[i..].len() >= 3 && alt[i..i + 3] == ["creative", "commons", "attribution"] {
            a.push("cc".into());
            a.push("by".into());
            i += 3;
        } else {
            a.push(alt[i].clone());
            i += 1;
        }
    }
    let a: Vec<&str> = a
        .iter()
        .map(String::as_str)
        .filter(|w| !is_filler(w))
        .collect();
    // Walk the words: take the longest run that is a known id, skip release words, fail on anything else. More than one id in a row is
    // fine ("MIT, Apache-2.0" has no "or" in it once the comma is dropped); at least one id is required.
    let (mut pos, mut ids) = (0, 0);
    while pos < a.len() {
        if let Some(n) = (1..=a.len() - pos)
            .rev()
            .find(|n| PERMISSIVE.contains(&a[pos..pos + n].join("-").as_str()))
        {
            pos += n;
            ids += 1;
        } else if is_release_word(a[pos]) || is_scope_word(a[pos]) {
            pos += 1;
        } else {
            return false;
        }
    }
    ids > 0
}

/// Read a licence name. See the module documentation for the rules and what is deliberately left as [`NameClass::Other`].
pub(crate) fn classify(licence: &str) -> NameClass {
    let mut w = words(licence);
    // "Apache-2.0-or-later" is Apache-2.0: drop "or later" / "and later" so the `or` is not read as an alternative
    let mut i = 0;
    while i + 1 < w.len() {
        if matches!(w[i].as_str(), "or" | "and") && w[i + 1] == "later" {
            w.drain(i..i + 2);
        } else {
            i += 1;
        }
    }
    if restricts_commercial_use(&w) {
        return NameClass::NonCommercial;
    }
    let (restricting_scope, any_scope) = scope_words(&w);
    if restricting_scope {
        return NameClass::NonCommercial;
    }
    let alternatives: Vec<&[String]> = w
        .split(|x| x == "or" || x == "and")
        .filter(|a| !a.is_empty())
        .collect();
    if !alternatives.is_empty() && alternatives.iter().all(|a| is_permissive_alternative(a)) {
        return NameClass::Permissive;
    }
    if any_scope {
        // "research preview" with no permissive licence behind it: still a research release, so not commercial
        return NameClass::NonCommercial;
    }
    NameClass::Other
}

#[cfg(test)]
mod tests {
    use super::*;
    use NameClass::{NonCommercial, Other, Permissive};

    #[test]
    fn non_commercial_phrasings_in_several_forms() {
        for l in [
            "non-commercial research",
            "Non-Commercial",
            "noncommercial",
            "NonCommercial",
            "non commercial",
            "No commercial use",
            "Not commercial",
            "Not for commercial use",
            "not-for-commercial-use",
            "For non commercial use only",
            "Commercial use prohibited",
            "Commercial use is not permitted",
            "Commercial use forbidden",
            "Commercial usage disallowed",
            "Free for non-profit use",
            "nonprofit use only",
            "Not for profit",
            "Nicht kommerziell",
            "Nicht-kommerzielle Nutzung",
            "Keine kommerzielle Nutzung",
            "Usage non commercial",
            "Uso no comercial",
            "Uso non commerciale",
            "research-only",
            "Research use only",
            "Academic use only",
            "evaluation only",
            "Free for personal and research use",
            "For personal use",
            "Educational use only",
            "CC-BY-NC-4.0",
            "cc-by-nc-sa-4.0",
            "CC BY-NC-ND 4.0",
            "cc_by_nc_4.0",
            "FLUX.1-dev Non-Commercial License",
            "Stability AI Non Commercial Research Community License",
            "Apache-2.0 research use only",
            "MIT OR CC-BY-NC-4.0",
            "research preview",
        ] {
            assert_eq!(classify(l), NonCommercial, "{l}");
        }
    }

    #[test]
    fn well_known_permissive_licences_in_common_spellings() {
        for l in [
            "MIT",
            "mit",
            "MIT License",
            "The MIT License",
            "Apache-2.0",
            "apache 2.0",
            "Apache License 2.0",
            "Apache License, Version 2.0",
            "Apache License v2.0",
            "Apache Software License 2.0",
            "apache2",
            "Apache-2.0-or-later",
            "BSD",
            "BSD-2-Clause",
            "BSD 2-Clause License",
            "BSD-3-Clause",
            "BSD 3-Clause License",
            "ISC",
            "ISC License",
            "CC0-1.0",
            "CC0 1.0 Universal",
            "Unlicense",
            "The Unlicense",
            "Zlib",
            "CC-BY-4.0",
            "CC BY 4.0",
            "cc-by-3.0",
            "Creative Commons Attribution 4.0 International",
            "Apache-2.0 OR MIT",
            "Apache-2.0 or MIT",
            "MIT/Apache-2.0",
            "MIT, Apache-2.0",
            "MIT AND BSD-3-Clause",
            "(MIT OR Apache-2.0)",
        ] {
            assert_eq!(classify(l), Permissive, "{l}");
        }
    }

    #[test]
    fn copyleft_rail_and_community_licences_are_left_unrecognised_on_purpose() {
        for l in [
            "GPL-3.0",
            "GPL-3.0-or-later",
            "GNU General Public License v3",
            "LGPL-2.1",
            "AGPL-3.0",
            "MPL-2.0",
            "CC-BY-SA-4.0",
            "openrail++",
            "CreativeML OpenRAIL-M",
            "OpenRAIL-M",
            "bigscience-bloom-rail-1.0",
            "llama3 community license",
            "Gemma Terms of Use",
            "proprietary",
            "other",
            "custom licence v3",
            "see LICENSE file",
            "MIT AND GPL-3.0",
            "Apache-2.0 WITH LLVM-exception",
            "",
            "   ",
        ] {
            assert_eq!(classify(l), Other, "{l}");
        }
    }

    #[test]
    fn a_word_that_describes_the_release_does_not_change_a_permissive_licence() {
        // The licence id decides; "research preview" and "personal fork" say what the release is, not what the terms allow.
        assert_eq!(classify("Apache-2.0 (research preview)"), Permissive);
        assert_eq!(classify("MIT personal fork"), Permissive);
        // but a scope phrase is a restriction, even next to a permissive id
        assert_eq!(classify("MIT research"), NonCommercial);
        assert_eq!(classify("Apache-2.0 for personal use"), NonCommercial);
        assert_eq!(classify("MIT (academic)"), NonCommercial);
        // and with no permissive id behind it, a research release is not commercial
        assert_eq!(classify("custom research preview"), NonCommercial);
    }

    #[test]
    fn nc_is_a_word_not_letters_inside_another_word() {
        assert_eq!(classify("Apache-2.0 (Lancaster fork)"), Other);
        assert_eq!(classify("Funcsoft licence"), Other);
    }
}
