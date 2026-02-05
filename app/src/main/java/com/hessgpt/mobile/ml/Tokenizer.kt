package com.hessgpt.mobile.ml

import android.content.Context
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import timber.log.Timber
import java.io.InputStreamReader

/**
 * Tokenizer pour HessGPT
 * Supporte BPE (Byte Pair Encoding) comme GPT-2/GPT-3
 */
class Tokenizer(
    private val context: Context,
    private val tokenizerFileName: String
) {
    
    private var vocab: Map<String, Int> = emptyMap()
    private var reverseVocab: Map<Int, String> = emptyMap()
    private var merges: List<Pair<String, String>> = emptyList()
    
    val vocabSize: Int
        get() = vocab.size
    
    val eosTokenId: Int
        get() = vocab["<|endoftext|>"] ?: vocab["</s>"] ?: 0
    
    val bosTokenId: Int
        get() = vocab["<|startoftext|>"] ?: vocab["<s>"] ?: 0
    
    private val padTokenId: Int
        get() = vocab["<|pad|>"] ?: eosTokenId
    
    /**
     * Charge le tokenizer depuis le fichier JSON
     */
    fun load() {
        try {
            val reader = InputStreamReader(context.assets.open(tokenizerFileName))
            val tokenizerData = Gson().fromJson(reader, TokenizerData::class.java)
            
            vocab = tokenizerData.vocab
            reverseVocab = vocab.entries.associate { it.value to it.key }
            merges = tokenizerData.merges?.map { 
                val parts = it.split(" ")
                parts[0] to parts[1]
            } ?: emptyList()
            
            Timber.d("Tokenizer chargé: ${vocab.size} tokens, ${merges.size} merges")
        } catch (e: Exception) {
            Timber.e(e, "Erreur lors du chargement du tokenizer")
            throw e
        }
    }
    
    /**
     * Encode un texte en IDs de tokens
     */
    fun encode(text: String): List<Int> {
        if (vocab.isEmpty()) {
            throw IllegalStateException("Le tokenizer n'est pas chargé!")
        }
        
        // Tokenization simple par caractères ou mots
        // Dans une vraie implémentation, utilisez BPE
        val tokens = tokenizeSimple(text)
        
        return tokens.mapNotNull { token ->
            vocab[token] ?: vocab["<|unk|>"] ?: run {
                Timber.w("Token inconnu: $token")
                null
            }
        }
    }
    
    /**
     * Décode des IDs de tokens en texte
     */
    fun decode(tokenIds: List<Int>): String {
        if (reverseVocab.isEmpty()) {
            throw IllegalStateException("Le tokenizer n'est pas chargé!")
        }
        
        val tokens = tokenIds.mapNotNull { id ->
            reverseVocab[id] ?: run {
                Timber.w("ID de token inconnu: $id")
                null
            }
        }
        
        return detokenize(tokens)
    }
    
    /**
     * Tokenization simple (à améliorer avec BPE réel)
     * Pour une vraie implémentation, utilisez une bibliothèque comme SentencePiece
     */
    private fun tokenizeSimple(text: String): List<String> {
        // Implémentation basique - séparer par espaces et ponctuation
        val normalized = text.trim()
        
        // Pattern de tokenization simple
        val pattern = Regex("""(\w+|[^\w\s])""")
        return pattern.findAll(normalized)
            .map { it.value }
            .toList()
    }
    
    /**
     * Détokenization - reconstituer le texte
     */
    private fun detokenize(tokens: List<String>): String {
        return tokens.joinToString(" ")
            .replace(" ,", ",")
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" '", "'")
            .replace("< |", "<|")
            .replace("| >", "|>")
    }
    
    /**
     * Classe de données pour parser le JSON du tokenizer
     */
    private data class TokenizerData(
        @SerializedName("vocab")
        val vocab: Map<String, Int>,
        
        @SerializedName("merges")
        val merges: List<String>? = null,
        
        @SerializedName("model_max_length")
        val modelMaxLength: Int? = 512,
        
        @SerializedName("added_tokens")
        val addedTokens: List<AddedToken>? = null
    )
    
    private data class AddedToken(
        @SerializedName("content")
        val content: String,
        
        @SerializedName("id")
        val id: Int
    )
}
