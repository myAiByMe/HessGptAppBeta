package com.hessgpt.mobile.ui

import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.hessgpt.mobile.R
import com.hessgpt.mobile.data.Message
import com.hessgpt.mobile.ml.ModelManager
import com.hessgpt.mobile.utils.PerformanceMonitor
import kotlinx.coroutines.launch
import timber.log.Timber

/**
 * Activité principale du chat avec HessGPT
 */
class ChatActivity : AppCompatActivity() {
    
    private lateinit var modelManager: ModelManager
    private lateinit var performanceMonitor: PerformanceMonitor
    
    private lateinit var recyclerView: RecyclerView
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var inputEditText: EditText
    private lateinit var sendButton: ImageButton
    private lateinit var loadingProgress: ProgressBar
    private lateinit var statusText: TextView
    
    private val messages = mutableListOf<Message>()
    private var isGenerating = false
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_chat)
        
        // Initialiser Timber pour le logging
        Timber.plant(Timber.DebugTree())
        
        initViews()
        setupRecyclerView()
        
        modelManager = ModelManager(this)
        performanceMonitor = PerformanceMonitor()
        
        loadModel()
        setupClickListeners()
    }
    
    private fun initViews() {
        recyclerView = findViewById(R.id.recyclerView)
        inputEditText = findViewById(R.id.inputEditText)
        sendButton = findViewById(R.id.sendButton)
        loadingProgress = findViewById(R.id.loadingProgress)
        statusText = findViewById(R.id.statusText)
    }
    
    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter(messages)
        recyclerView.apply {
            layoutManager = LinearLayoutManager(this@ChatActivity).apply {
                stackFromEnd = true
            }
            adapter = chatAdapter
        }
    }
    
    private fun loadModel() {
        showLoading("Chargement du modèle...")
        
        lifecycleScope.launch {
            try {
                val success = modelManager.loadModel()
                
                if (success) {
                    showStatus("Modèle chargé ✓", true)
                    addSystemMessage("HessGPT est prêt ! Posez-moi vos questions.")
                } else {
                    showStatus("Erreur de chargement", false)
                    Toast.makeText(
                        this@ChatActivity,
                        "Impossible de charger le modèle",
                        Toast.LENGTH_LONG
                    ).show()
                }
            } catch (e: Exception) {
                Timber.e(e, "Erreur lors du chargement")
                showStatus("Erreur", false)
                Toast.makeText(
                    this@ChatActivity,
                    "Erreur: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
    
    private fun setupClickListeners() {
        sendButton.setOnClickListener {
            val text = inputEditText.text.toString().trim()
            if (text.isNotEmpty() && !isGenerating) {
                sendMessage(text)
                inputEditText.text.clear()
            }
        }
    }
    
    private fun sendMessage(text: String) {
        // Ajouter le message de l'utilisateur
        val userMessage = Message(
            text = text,
            isUser = true,
            timestamp = System.currentTimeMillis()
        )
        addMessage(userMessage)
        
        // Préparer le message de l'assistant
        val assistantMessage = Message(
            text = "",
            isUser = false,
            timestamp = System.currentTimeMillis()
        )
        val messageIndex = addMessage(assistantMessage)
        
        // Générer la réponse
        generateResponse(text, messageIndex)
    }
    
    private fun generateResponse(prompt: String, messageIndex: Int) {
        isGenerating = true
        updateSendButtonState()
        showStatus("Génération...", true)
        
        lifecycleScope.launch {
            try {
                performanceMonitor.startGeneration()
                
                val fullResponse = StringBuilder()
                
                modelManager.generate(
                    prompt = prompt,
                    maxNewTokens = 150,
                    onTokenGenerated = { token ->
                        // Streaming: mettre à jour le message en temps réel
                        fullResponse.append(token)
                        updateMessage(messageIndex, fullResponse.toString())
                        performanceMonitor.recordToken()
                        
                        // Scroller vers le bas
                        recyclerView.smoothScrollToPosition(messages.size - 1)
                    }
                )
                
                performanceMonitor.endGeneration()
                
                // Afficher les statistiques de performance
                val stats = performanceMonitor.getStats()
                Timber.d("Performance: ${stats.tokensPerSecond} tokens/sec, " +
                        "latence: ${stats.latencyMs}ms")
                
                showStatus(
                    "✓ ${stats.tokensPerSecond} tok/s | ${stats.totalTokens} tokens",
                    true
                )
                
            } catch (e: Exception) {
                Timber.e(e, "Erreur lors de la génération")
                updateMessage(
                    messageIndex,
                    "Désolé, une erreur s'est produite: ${e.message}"
                )
                showStatus("Erreur", false)
            } finally {
                isGenerating = false
                updateSendButtonState()
            }
        }
    }
    
    private fun addMessage(message: Message): Int {
        messages.add(message)
        chatAdapter.notifyItemInserted(messages.size - 1)
        recyclerView.smoothScrollToPosition(messages.size - 1)
        return messages.size - 1
    }
    
    private fun updateMessage(index: Int, text: String) {
        if (index >= 0 && index < messages.size) {
            messages[index].text = text
            chatAdapter.notifyItemChanged(index)
        }
    }
    
    private fun addSystemMessage(text: String) {
        val message = Message(
            text = text,
            isUser = false,
            timestamp = System.currentTimeMillis(),
            isSystem = true
        )
        addMessage(message)
    }
    
    private fun showLoading(message: String) {
        loadingProgress.visibility = View.VISIBLE
        statusText.text = message
    }
    
    private fun showStatus(message: String, success: Boolean) {
        loadingProgress.visibility = View.GONE
        statusText.text = message
        statusText.setTextColor(
            if (success) 
                getColor(android.R.color.holo_green_dark)
            else 
                getColor(android.R.color.holo_red_dark)
        )
    }
    
    private fun updateSendButtonState() {
        sendButton.isEnabled = !isGenerating && modelManager.isModelLoaded()
        sendButton.alpha = if (sendButton.isEnabled) 1.0f else 0.5f
    }
    
    override fun onDestroy() {
        super.onDestroy()
        modelManager.release()
    }
}
