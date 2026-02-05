package com.hessgpt.mobile.ml

import android.content.Context
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Gestionnaire du modèle HessGPT avec PyTorch Mobile
 * Gère le chargement, l'inférence et la libération du modèle
 */
class ModelManager(private val context: Context) {
    
    private var module: Module? = null
    private var tokenizer: Tokenizer? = null
    
    // Configuration du modèle
    private val modelFileName = "model.ptl"
    private val tokenizerFileName = "tokenizer.json"
    
    // Hyperparamètres
    private val maxSeqLength = 512
    private val temperature = 0.7f
    private val topK = 40
    private val topP = 0.9f
    
    /**
     * Charge le modèle depuis les assets
     */
    suspend fun loadModel(): Boolean = withContext(Dispatchers.IO) {
        try {
            Timber.d("Début du chargement du modèle...")
            
            // Copier le modèle depuis assets vers le cache
            val modelFile = assetFilePath(modelFileName)
            
            // Charger le module PyTorch
            module = Module.load(modelFile)
            
            // Charger le tokenizer
            tokenizer = Tokenizer(context, tokenizerFileName)
            tokenizer?.load()
            
            Timber.d("Modèle chargé avec succès!")
            true
        } catch (e: Exception) {
            Timber.e(e, "Erreur lors du chargement du modèle")
            false
        }
    }
    
    /**
     * Génère une réponse à partir d'un prompt
     * @param prompt Le texte d'entrée
     * @param maxNewTokens Nombre maximum de tokens à générer
     * @param onTokenGenerated Callback appelé à chaque token généré (streaming)
     */
    suspend fun generate(
        prompt: String,
        maxNewTokens: Int = 100,
        onTokenGenerated: ((String) -> Unit)? = null
    ): String = withContext(Dispatchers.Default) {
        
        if (module == null || tokenizer == null) {
            throw IllegalStateException("Le modèle n'est pas chargé!")
        }
        
        try {
            // Tokenize l'entrée
            val inputIds = tokenizer!!.encode(prompt)
            Timber.d("Input tokenizé: ${inputIds.size} tokens")
            
            val generatedTokens = mutableListOf<Int>()
            generatedTokens.addAll(inputIds)
            
            var currentSequence = inputIds.toIntArray()
            
            // Générer token par token
            repeat(maxNewTokens) { step ->
                
                // Créer le tensor d'entrée
                val inputTensor = Tensor.fromBlob(
                    currentSequence.map { it.toLong() }.toLongArray(),
                    longArrayOf(1, currentSequence.size.toLong())
                )
                
                // Forward pass
                val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
                
                // Extraire les logits du dernier token
                val logits = outputTensor.dataAsFloatArray
                val vocabSize = tokenizer!!.vocabSize
                
                // Prendre les logits du dernier token
                val lastTokenLogits = logits.takeLast(vocabSize).toFloatArray()
                
                // Appliquer temperature
                val scaledLogits = lastTokenLogits.map { it / temperature }.toFloatArray()
                
                // Sampling avec top-k et top-p
                val nextToken = sample(scaledLogits, topK, topP)
                
                // Vérifier le token de fin
                if (nextToken == tokenizer!!.eosTokenId) {
                    Timber.d("Token EOS détecté, arrêt de la génération")
                    break
                }
                
                generatedTokens.add(nextToken)
                
                // Mettre à jour la séquence courante (avec KV-cache simulé)
                currentSequence = generatedTokens.takeLast(maxSeqLength).toIntArray()
                
                // Callback streaming
                onTokenGenerated?.let {
                    val decodedToken = tokenizer!!.decode(listOf(nextToken))
                    withContext(Dispatchers.Main) {
                        it(decodedToken)
                    }
                }
                
                if (step % 10 == 0) {
                    Timber.d("Génération: $step tokens")
                }
            }
            
            // Décoder tous les tokens générés (seulement les nouveaux)
            val newTokens = generatedTokens.drop(inputIds.size)
            val result = tokenizer!!.decode(newTokens)
            
            Timber.d("Génération terminée: ${newTokens.size} tokens")
            result
            
        } catch (e: Exception) {
            Timber.e(e, "Erreur lors de la génération")
            throw e
        }
    }
    
    /**
     * Sampling avec top-k et top-p (nucleus sampling)
     */
    private fun sample(logits: FloatArray, topK: Int, topP: Float): Int {
        // Convertir logits en probabilités avec softmax
        val maxLogit = logits.maxOrNull() ?: 0f
        val expLogits = logits.map { Math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExp = expLogits.sum()
        val probs = expLogits.map { it / sumExp }
        
        // Top-K filtering
        val sortedIndices = probs.indices.sortedByDescending { probs[it] }
        val topKIndices = sortedIndices.take(topK)
        
        // Top-P (nucleus) filtering
        var cumulativeProb = 0f
        val nucleusIndices = mutableListOf<Int>()
        
        for (idx in topKIndices) {
            nucleusIndices.add(idx)
            cumulativeProb += probs[idx]
            if (cumulativeProb >= topP) break
        }
        
        // Renormaliser les probabilités
        val nucleusProbs = nucleusIndices.map { probs[it] }
        val sumNucleusProbs = nucleusProbs.sum()
        val normalizedProbs = nucleusProbs.map { it / sumNucleusProbs }
        
        // Sampling multinomial
        val random = Math.random().toFloat()
        var cumProb = 0f
        
        for (i in normalizedProbs.indices) {
            cumProb += normalizedProbs[i]
            if (random <= cumProb) {
                return nucleusIndices[i]
            }
        }
        
        return nucleusIndices.last()
    }
    
    /**
     * Copie un fichier depuis assets vers le cache
     */
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        
        if (file.exists()) {
            Timber.d("Fichier existe déjà dans le cache: ${file.absolutePath}")
            return file.absolutePath
        }
        
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        
        Timber.d("Fichier copié dans le cache: ${file.absolutePath}")
        return file.absolutePath
    }
    
    /**
     * Libère les ressources du modèle
     */
    fun release() {
        module?.destroy()
        module = null
        tokenizer = null
        Timber.d("Ressources du modèle libérées")
    }
    
    /**
     * Vérifie si le modèle est chargé
     */
    fun isModelLoaded(): Boolean = module != null && tokenizer != null
}
